"""Proof search using best-first search.
"""

import os
import sys
import ray
import time
import heapq
import torch
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    LeanError,
    TimeoutError,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoHardTimeoutError,
)
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ray.util.actor_pool import ActorPool

from common import zip_strict
from prover.search_tree import *
from generator.model import RetrievalAugmentedGenerator, FixedTacticGenerator


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: Optional[List[str]]

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int


class BestFirstSearchProver:
    """A prover that uses best-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.posision = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        if isinstance(self.tac_gen, FixedTacticGenerator):
            imps = [self.tac_gen.module]
        else:
            imps = []

        try:
            with Dojo(thm, hard_timeout=60 + self.timeout, additional_imports=imps) as (
                dojo,
                init_state,
            ):
                self.dojo = dojo
                self.root = InternalNode(
                    state=init_state,
                    cumulative_logprob=0.0,
                )
                self.nodes = {init_state: self.root}
                self.priority_queue = [self.root]

                with torch.no_grad():
                    try:
                        self._best_first_search()
                    except DojoCrashError as ex:
                        logger.warning(f"Dojo crashed with {ex} when proving {thm}")
                        pass

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def _best_first_search(self) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            try:
                self._step()
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

    def _step(self):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority.
        search_node = heapq.heappop(self.priority_queue)
        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
        else:
            ts = search_node.state.unsolved_tactic_state
        suggestions = self._generate_tactics(ts)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = []
        for tactic, logprob in suggestions:
            edge, finished = self._run_tactic(search_node, tactic, logprob)
            results.append(edge)
            if finished:
                break

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants()

    def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)

        if self.theorem.repo != self.repo:
            path = self.theorem.repo.get_packages_dir() / self.theorem.repo.name / path

        suggestions = self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.posision,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(
        self, node: InternalNode, tactic: str, logprob: float
    ) -> Tuple[Edge, bool]:
        t0 = time.monotonic()
        response = self.dojo.run_tac(node.state, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        try:
            # If we've seen this response before, use the existing node
            result_node = self.nodes[response]
        except KeyError:
            # Build a new node
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        # Record the new node and add it to the search queue.
        self.nodes[response] = result_node

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result_node)

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge, isinstance(response, ProofFinished)

    #########
    # DEBUG #
    #########

    def check_invariants(self):
        """Perform some sanity checks."""
        for node in self.priority_queue:
            assert node in self.nodes.values()
            assert isinstance(node, InternalNode)
            assert not node.is_explored

        for response, node in self.nodes.items():
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofFinishedNode)
                assert node not in self.priority_queue
                assert self.root.status == Status.PROVED
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                assert isinstance(node, ErrorNode)
                assert node not in self.priority_queue
            else:
                assert isinstance(node, InternalNode)

                if node.is_explored:
                    assert node not in self.priority_queue
                else:
                    assert node in self.priority_queue

                node.check_invariants()

# TODO: later ensure this has correct stuff like config dict
@ray.remote
class CpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a CPU."""

    def __init__(
        self,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        if ckpt_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        else:
            tac_gen = RetrievalAugmentedGenerator.load(
                ckpt_path, device=torch.device("cpu"), freeze=True
            )
            if tac_gen.retriever is not None:
                if indexed_corpus_path is not None:
                    tac_gen.retriever.load_corpus(indexed_corpus_path)
                tac_gen.retriever.reindex_corpus(batch_size=32)
        super().__init__(
            tac_gen,
            timeout,
            num_sampled_tactics,
            debug,
        )

# TODO: reduce repetition with main

def find_latest_checkpoint(raid_dir, checkpoint_dir):
    """Finds the most recent checkpoint."""
    checkpoint_dir = raid_dir + "/" + checkpoint_dir
    all_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not all_checkpoints:
        raise FileNotFoundError("No checkpoints found.")
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    logger.info(f"Using the latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

@ray.remote(num_gpus=1)
class GpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a GPU."""

    def __init__(
        self,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        raid_dir: str,
        checkpoint_dir: str,
        debug: bool,
        run_progressive_training: bool = True,
    ) -> None:
    
        if ckpt_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        else:
            model_checkpoint_path = None
            if run_progressive_training:
                model_checkpoint_path = find_latest_checkpoint(raid_dir, checkpoint_dir)
            else:
                model_checkpoint_path = "/raid/adarsh/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
            
            config = {
                "model_name": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
                "lr": 1e-3,
                "warmup_steps": 1000,
                "num_beams": 5,
                "eval_num_retrieved": 10,
                "eval_num_workers": 1,
                "eval_num_gpus": 1,
                "eval_num_theorems": 100,
                "max_inp_seq_len": 512,
                "max_oup_seq_len": 128,
                "ret_ckpt_path": model_checkpoint_path,
            }

            tac_gen = RetrievalAugmentedGenerator.load(
                ckpt_path, device=torch.device("cuda"), freeze=True, config=config
            )
            logger.info(f"Loaded model from {ckpt_path}")
            logger.info(f"Using retriever: {tac_gen.retriever}")
            if tac_gen.retriever is not None:
                if indexed_corpus_path is not None:
                    logger.info(f"Loading indexed corpus from {indexed_corpus_path}")
                    tac_gen.retriever.load_corpus(indexed_corpus_path)
                    logger.info(f"Loaded indexed corpus from {indexed_corpus_path}")
                tac_gen.retriever.reindex_corpus(batch_size=32)
                logger.info("Finished reindexing!")
        super().__init__(
            tac_gen,
            timeout,
            num_sampled_tactics,
            debug,
        )

class DistributedProver:
    """A distributed prover that uses Ray to parallelize the proof search.

    It is a wrapper around `CpuProver` and `GpuProver` that handles the different
    devices and different number of concurrent provers.
    """

    def __init__(
        self,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        num_workers: int,
        num_gpus: int,
        timeout: int,
        num_sampled_tactics: int,
        raid_dir: str,
        checkpoint_dir: str,
        debug: Optional[bool] = False,
        run_progressive_training: bool = True,
    ) -> None:
        if ckpt_path is None:
            assert tactic and not indexed_corpus_path
        else:
            assert not tactic and not module
        self.distributed = num_workers > 1

        if not self.distributed:
            if ckpt_path is None:
                tac_gen = FixedTacticGenerator(tactic, module)
            else:
                device = torch.device("cuda") if num_gpus > 0 else torch.device("cpu")

                model_checkpoint_path = None
                if run_progressive_training:
                    model_checkpoint_path = find_latest_checkpoint(raid_dir, checkpoint_dir)
                else:
                    model_checkpoint_path = "/raid/adarsh/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                
                config = {
                    "model_name": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
                    "lr": 1e-3,
                    "warmup_steps": 1000,
                    "num_beams": 5,
                    "eval_num_retrieved": 10,
                    "eval_num_workers": 1,
                    "eval_num_gpus": 1,  # TODO: change for GPU
                    "eval_num_theorems": 100,
                    "max_inp_seq_len": 512,
                    "max_oup_seq_len": 128,
                    "ret_ckpt_path": model_checkpoint_path,
                }
                tac_gen = RetrievalAugmentedGenerator.load(
                    ckpt_path, device=device, freeze=True, config=config
                )
                logger.info(f"Loaded model from {ckpt_path}")
                logger.info(f"Using retriever: {tac_gen.retriever}")
                if tac_gen.retriever is not None:
                    if indexed_corpus_path is not None:
                        logger.info(f"Loading indexed corpus from {indexed_corpus_path}")
                        tac_gen.retriever.load_corpus(indexed_corpus_path)
                        logger.info(f"Loaded indexed corpus from {indexed_corpus_path}")
                    tac_gen.retriever.reindex_corpus(batch_size=32)
                    logger.info("Finished reindexing!")
            self.prover = BestFirstSearchProver(
                tac_gen, timeout, num_sampled_tactics, debug
            )
            return

        if num_gpus >= 1:
            logger.info(f"Launching {num_workers} workers with {num_gpus} GPUs.")
            num_gpus_per_worker = num_gpus / num_workers
            provers = [
                GpuProver.options(num_gpus=num_gpus_per_worker).remote(
                    ckpt_path,
                    indexed_corpus_path,
                    tactic,
                    module,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for _ in range(num_workers)
            ]
        else:
            logger.info(f"Launching {num_workers} CPU workers.")
            provers = [
                CpuProver.remote(
                    ckpt_path,
                    indexed_corpus_path,
                    tactic,
                    module,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for _ in range(num_workers)
            ]

        self.prover_pool = ActorPool(provers)

    def search_unordered(self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]) -> List[Optional[SearchResult]]:
        results = []
        for i, (thm, pos) in enumerate(zip_strict(theorems, positions)):
            try:
                if not self.distributed:
                    logger.info(f"Not distributed")
                    result = self.prover.search(repo, thm, pos)
                    logger.info(f"Not distributed, finished")
                else:
                    # TODO: fix this later
                    logger.info(f"Distributed")
                    result = ray.get(self.prover_pool.submit(lambda p, args: p.search.remote(*args), (repo, thm, pos)))
                    logger.info(f"Distributed, finished")
                results.append(result)
                logger.info(f"Completed theorem: {thm.full_name}")
                logger.info(f"Result: {result}")
            except Exception as e:
                logger.error(f"Error processing theorem {thm.full_name}: {str(e)}")
                results.append(None)
        logger.info(f"Completed search_unordered")
        return results