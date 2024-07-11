"""Ligihtning module for the premise retriever."""

import os
import json
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
from lean_dojo import Pos
from loguru import logger
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Union
from transformers import T5EncoderModel, AutoTokenizer
from torch.distributed import barrier

from common import (
    Premise,
    Context,
    Corpus,
    get_optimizers,
    load_checkpoint,
    zip_strict,
    cpu_checkpointing_enabled,
)


torch.set_float32_matmul_precision("medium")


class PremiseRetriever(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        max_seq_len: int,
        num_retrieved: int = 100,
    ) -> None:
        # logger.info("Inside __init__")
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.embeddings_staled = True
        self.train_loss = []
        self.previous_params = {}
        self.fisher_info = {}
        # logger.info("End of __init__")

    def compute_fisher_information(self, dataloader):
        logger.info("Computing Fisher Information")
        # TODO: compute at end of running
        # TODO: save the matrix
        # TODO: do on 4 GPUs
        # TODO: use old dataloader
        if not torch.cuda.is_available():
            logger.warning("Indexing the corpus using CPU can be very slow.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        # Code to compute the Fisher Information Matrix after training on the first task
        self.eval()
        # Use tqdm to see progress
        for batch in tqdm(dataloader):
            self.zero_grad()
            # output = self.forward(**batch)
            # loss = self.loss_function(output, batch['labels'])
            context_ids, context_mask = batch["context_ids"].to(device), batch["context_mask"].to(device)
            pos_premise_ids, pos_premise_mask = batch["pos_premise_ids"].to(device), batch["pos_premise_mask"].to(device)
            neg_premises_ids, neg_premises_mask = [x.to(device) for x in  batch["neg_premises_ids"]], [x.to(device) for x in batch["neg_premises_mask"]]
            label = batch["label"].to(device)
            loss = self.forward(
                context_ids,
                context_mask,
                pos_premise_ids,
                pos_premise_mask,
                neg_premises_ids,
                neg_premises_mask,
                label,
            )
            loss.backward()
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] = param.grad ** 2 + self.fisher_info.get(name, 0)

        return self.fisher_info

    def ewc_loss(self, lamda=5000):
        # TODO: choose lambda and ewc_loss better if needed
        ewc_loss = 0
        for name, param in self.named_parameters():
            if name in self.previous_params:
                # EWC Penalty is the sum of the squares of differences times the Fisher Information
                ewc_loss += (self.fisher_info[name] * (param - self.previous_params[name]) ** 2).sum()
        return lamda * ewc_loss

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool, config: dict) -> "PremiseRetriever":
        # logger.info("Inside load")
        # logger.info("End of load")
        return load_checkpoint(cls, ckpt_path, device, freeze, config)

    def load_corpus(self, path_or_corpus: Union[str, Corpus]) -> None:
        # logger.info("Inside load_corpus")
        """Associate the retriever with a corpus."""
        if isinstance(path_or_corpus, Corpus):
            self.corpus = path_or_corpus
            self.corpus_embeddings = None
            self.embeddings_staled = True
            # logger.info("End of load_corpus inside if")
            return

        path = path_or_corpus
        if path.endswith(".jsonl"):  # A raw corpus without embeddings.
            self.corpus = Corpus(path)
            self.corpus_embeddings = None
            self.embeddings_staled = True
        else:  # A corpus with pre-computed embeddings.
            indexed_corpus = pickle.load(open(path, "rb"))
            self.corpus = indexed_corpus.corpus
            self.corpus_embeddings = indexed_corpus.embeddings
            self.embeddings_staled = False
        # logger.info("End of load_corpus outside if")

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        # logger.info("Inside embedding_size")
        # logger.info("End of embedding_size")
        return self.encoder.config.hidden_size

    def _encode(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        if cpu_checkpointing_enabled(self):
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.encoder, input_ids, attention_mask, use_reentrant=False
            )[0]
        else:
            hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        # Masked average.
        lens = attention_mask.sum(dim=1)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        return F.normalize(features, dim=1)

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_mask: torch.LongTensor,
        pos_premise_ids: torch.LongTensor,
        pos_premise_mask: torch.LongTensor,
        neg_premises_ids: torch.LongTensor,
        neg_premises_mask: torch.LongTensor,
        label: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        # logger.info("Inside forward")
        # Encode the query and positive/negative documents.
        context_emb = self._encode(context_ids, context_mask)
        pos_premise_emb = self._encode(pos_premise_ids, pos_premise_mask)
        neg_premise_embs = [
            self._encode(ids, mask)
            for ids, mask in zip_strict(neg_premises_ids, neg_premises_mask)
        ]
        all_premise_embs = torch.cat([pos_premise_emb, *neg_premise_embs], dim=0)

        # Cosine similarities for unit-norm vectors are just inner products.
        similarity = torch.mm(context_emb, all_premise_embs.t())
        assert -1 <= similarity.min() <= similarity.max() <= 1
        loss = F.mse_loss(similarity, label)
        # logger.info("End of forward")
        return loss

    ############
    # Training #
    ############

    def on_fit_start(self) -> None:
        logger.info("Inside on_fit_start")
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            logger.info(f"Logging to {self.trainer.log_dir}")

        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True
        logger.info("End of on_fit_start")

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        # logger.info("Inside training_step")
        loss = self(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )
        # logger.info(f"Training loss before EWC: {loss.item():.4f}")
        # loss += self.ewc_loss()
        # logger.info(f"Training loss after EWC: {loss.item():.4f}")
        self.train_loss.append(loss.item())
        self.log(
            "loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )
        # logger.info("End of training_step")
        return loss

    def on_train_batch_end(self, outputs, batch, _) -> None:
        """Mark the embeddings as staled after a training batch."""
        # logger.info("Inside on_train_batch_end")
        self.embeddings_staled = True
        # logger.info("End of on_train_batch_end")

    def configure_optimizers(self) -> Dict[str, Any]:
        # logger.info("Inside configure_optimizers")
        # logger.info("End of configure_optimizers")
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    ##############
    # Validation #
    ##############

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        # logger.info("Inside reindex_corpus")
        """Re-index the retrieval corpus using the up-to-date encoder."""
        if not self.embeddings_staled:
            return
        logger.info("Re-indexing the retrieval corpus")

        self.corpus_embeddings = torch.zeros(
            len(self.corpus.all_premises),
            self.embedding_size,
            dtype=self.encoder.dtype,
            device=self.device,
        )

        for i in tqdm(range(0, len(self.corpus), batch_size)):
            batch_premises = self.corpus.all_premises[i : i + batch_size]
            tokenized_premises = self.tokenizer(
                [p.serialize() for p in batch_premises],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.corpus_embeddings[i : i + batch_size] = self._encode(
                tokenized_premises.input_ids, tokenized_premises.attention_mask
            )
        self.embeddings_staled = False
        # logger.info("End of reindex_corpus")

    def on_validation_start(self) -> None:
        # logger.info("Inside on_validation_start")
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)
        # logger.info("End of on_validation_start")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate metrics such as Recall@K and MRR."""
        logger.info("Inside validation_step")
        logger.info("All training loss for epoch", self.train_loss)
        # Retrieval.
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, _ = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        num_with_premises = 0
        tb = self.logger.experiment

        for i, (all_pos_premises, premises) in enumerate(
            zip_strict(batch["all_pos_premises"], retrieved_premises)
        ):
            # Only log the first example in the batch.
            if i == 0:
                msg_gt = "\n\n".join([p.serialize() for p in all_pos_premises])
                msg_retrieved = "\n\n".join(
                    [f"{j}. {p.serialize()}" for j, p in enumerate(premises)]
                )
                TP = len(set(premises).intersection(all_pos_premises))
                if len(all_pos_premises) == 0:
                    r = math.nan
                else:
                    r = float(TP) / len(all_pos_premises)
                msg = f"Recall@{self.num_retrieved}: {r}\n\nGround truth:\n\n```\n{msg_gt}\n```\n\nRetrieved:\n\n```\n{msg_retrieved}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            all_pos_premises = set(all_pos_premises)
            if len(all_pos_premises) == 0:
                continue
            else:
                num_with_premises += 1
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premises))
                if premises[j] in all_pos_premises and not first_match_found:
                    MRR.append(1.0 / (j + 1))
                    first_match_found = True
            if not first_match_found:
                MRR.append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            logger.info(f"Recall@{j+1}_val: {recall[j]}")
            self.log(
                f"Recall@{j+1}_val",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=num_with_premises,
            )

        logger.info(f"MRR: {np.mean(MRR)}")
        self.log(
            "MRR",
            np.mean(MRR),
            on_epoch=True,
            sync_dist=True,
            batch_size=num_with_premises,
        )
        logger.info("End of validation_step")

    ##############
    # Prediction #
    ##############

    def on_predict_start(self) -> None:
        # logger.info("Inside on_predict_start")
        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)
        # self.corpus_embeddings = torch.zeros(
        #     len(self.corpus.all_premises),
        #     self.embedding_size,
        #     dtype=self.encoder.dtype,
        #     device=self.device,
        # )
        # self.embeddings_staled = False
        self.predict_step_outputs = []
        # logger.info("End of on_predict_start")

    def predict_step(self, batch: Dict[str, Any], _):
        # logger.info("Inside predict_step")
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        for (
            url,
            commit,
            file_path,
            full_name,
            start,
            tactic_idx,
            ctx,
            pos_premises,
            premises,
            s,
        ) in zip_strict(
            batch["url"],
            batch["commit"],
            batch["file_path"],
            batch["full_name"],
            batch["start"],
            batch["tactic_idx"],
            batch["context"],
            batch["all_pos_premises"],
            retrieved_premises,
            scores,
        ):
            self.predict_step_outputs.append(
                {
                    "url": url,
                    "commit": commit,
                    "file_path": file_path,
                    "full_name": full_name,
                    "start": start,
                    "tactic_idx": tactic_idx,
                    "context": ctx,
                    "all_pos_premises": pos_premises,
                    "retrieved_premises": premises,
                    "scores": s,
                }
            )
        # logger.info("End of predict_step")
    
    # def _eval(self, data, preds_map) -> Tuple[float, float, float]:
    #     R1 = []
    #     R10 = []
    #     MRR = []

    #     for thm in tqdm(data):
    #         for i, _ in enumerate(thm["traced_tactics"]):
    #             logger.info(f"thm['file_path']: {thm['file_path']}")
    #             logger.info(f"thm['full_name']: {thm['full_name']}")
    #             logger.info(f"tuple(thm['start']): {tuple(thm['start'])}")
    #             logger.info(f"i: {i}")
    #             # pred = preds_map[
    #             #     (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
    #             # ]
    #             pred = None
    #             key = (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
    #             logger.info(f"Checking if key {key} is in preds_map")
    #             if key in preds_map:
    #                 pred = preds_map[key]
    #                 logger.info(f"Key {key} found in predictions")
    #             else:
    #                 logger.info(f"Key {key} not found in predictions.")
    #                 continue  # or handle as appropriate
    #             all_pos_premises = set(pred["all_pos_premises"])
    #             if len(all_pos_premises) == 0:
    #                 continue

    #             retrieved_premises = pred["retrieved_premises"]
    #             TP1 = retrieved_premises[0] in all_pos_premises
    #             R1.append(float(TP1) / len(all_pos_premises))
    #             TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
    #             R10.append(float(TP10) / len(all_pos_premises))

    #             for j, p in enumerate(retrieved_premises):
    #                 if p in all_pos_premises:
    #                     MRR.append(1.0 / (j + 1))
    #                     break
    #             else:
    #                 MRR.append(0.0)

    #     R1 = 100 * np.mean(R1)
    #     R10 = 100 * np.mean(R10)
    #     MRR = np.mean(MRR)
    #     return R1, R10, MRR

    def on_predict_epoch_end(self) -> None:
        # logger.info("Inside on_predict_epoch_end")
        if self.trainer.log_dir is not None:
            logger.info("About to construct predictions map")
            # logger.info("FULL DETAILS")
            # for p in self.predict_step_outputs:
            #     print(p)

            for p in self.predict_step_outputs:
                # if p["file_path"] == ".lake/packages/mathlib/Mathlib/Topology/Category/TopCat/Limits/Products.lean" or p["file_path"] == ".lake/packages/mathlib/Mathlib/Algebra/Order/Field/Basic.lean" or p["file_path"] == ".lake/packages/mathlib/Mathlib/Algebra/EuclideanDomain/Defs.lean":
                logger.info(f"p['file_path']: {p['file_path']}")
                logger.info(f"p['full_name']: {p['full_name']}")
                logger.info(f"tuple(p['start']): {tuple(p['start'])}")
                logger.info(f"p['tactic_idx']: {p['tactic_idx']}")

            gpu_id = self.trainer.local_rank
            
            preds_map = {
                (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
                for p in self.predict_step_outputs
            }

            # path = os.path.join(self.trainer.log_dir, "test_pickle.pkl")
            # with open(path, "wb") as oup:
            #     pickle.dump(self.predict_step_outputs, oup)
            # logger.info(f"Retrieval predictions saved to {path}")

            path = f"test_pickle_{gpu_id}.pkl"
            with open(path, "wb") as oup:
                pickle.dump(preds_map, oup)
            logger.info(f"Retrieval predictions saved to {path}")

            # path = "test_predictions.txt"
            # with open(path,'w+') as f:
            #     f.write(str(preds_map))
            # logger.info(f"Retrieval predictions saved to {path}")

            # Save to text file too
            # path = os.path.join(self.trainer.log_dir, "test_predictions.txt")
            # with open(path, "wb") as oup:
            #     oup.write(json.dumps(self.predict_step_outputs).encode("ascii"))
            # logger.info(f"Retrieval predictions saved to {path}")
            
            # preds_map = {
            #     (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
            #     for p in self.predict_step_outputs
            # }
            # assert len(self.predict_step_outputs) == len(preds_map), "Duplicate predictions found!"
            # curr_R1 = []
            # curr_R10 = []
            # curr_MRR = []
            # split = "test"
            # data_path = os.path.join(self.trainer.datamodule.data_path, f"{split}.json")
            # data = json.load(open(data_path))
            # logger.info(f"Evaluating on {data_path}")
            # R1, R10, MRR = self._eval(data, preds_map)
            # logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
            # curr_R1.append(R1)
            # curr_R10.append(R10)
            # curr_MRR.append(MRR)

            # # Save results to a text file
            # results_path = "evaluation_results.txt"
            # with open(results_path, 'w') as f:
            #     results = zip(curr_R1, curr_R10, curr_MRR)
            #     for r1, r10, mrr in results:
            #         f.write(f"R1: {r1}, R10: {r10}, MRR: {mrr}\n")

        self.predict_step_outputs.clear()
        barrier()

        if self.trainer.is_global_zero:
            logger.info("All GPUs have completed their predictions and saved the data.")
        # logger.info("End of on_predict_epoch_end")

    def retrieve(
        self,
        state: List[str],
        file_name: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        k: int,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        # logger.info("Inside retrieve")
        self.reindex_corpus(batch_size=32)

        ctx = [
            Context(*_)
            for _ in zip_strict(file_name, theorem_full_name, theorem_pos, state)
        ]
        ctx_tokens = self.tokenizer(
            [_.serialize() for _ in ctx],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        context_emb = self._encode(
            ctx_tokens.input_ids.to(self.device),
            ctx_tokens.attention_mask.to(self.device),
        )

        if self.corpus_embeddings.device != context_emb.device:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.device)
        if self.corpus_embeddings.dtype != context_emb.dtype:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.dtype)

        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            ctx,
            context_emb,
            k,
        )
        # logger.info("End of retrieve")
        return retrieved_premises, scores
