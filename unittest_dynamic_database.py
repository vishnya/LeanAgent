import unittest
import datetime
from pathlib import Path
from dynamic_database import DynamicDatabase, Repository, Theorem, AnnotatedTactic, Annotation, PremiseFile, Premise
from lean_dojo.data_extraction.lean import Pos, LeanGitRepo
import generate_benchmark_lean4
import lean_dojo
import json
import shutil
import random
from loguru import logger
from unittest.mock import Mock, patch
from dynamic_database import DynamicDatabase, Repository, Theorem, AnnotatedTactic
from prover.proof_search import Status, SearchResult
from main import prove_sorry_theorems, retrieve_proof

RAID_DIR = "/raid/adarsh"
DATA_DIR = "datasets_new"
MERGED_DATA_DIR = "datasets_merged"

class TestDynamicDatabaseUnicode(unittest.TestCase):
    def setUp(self):
        self.db = DynamicDatabase()
        self.unicode_repo = self.create_unicode_sample_repo()
        self.db.add_repository(self.unicode_repo)
    
    def create_unicode_sample_repo(self):
        repo = Repository(
            url="https://github.com/example/repo",
            name="Example Repo with Unicode ユニコード",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            date_processed=datetime.datetime.now(),
            metadata={"key": "value with Unicode ✨"},
        )

        theorem1 = Theorem(
            full_name="example.commutative_addition",
            theorem_statement="∀ x y : ℕ, x + y = y + x",
            file_path=Path("src/example.lean"),
            start=Pos(1, 1),
            end=Pos(5, 10),
            url="https://github.com/example/repo",
            commit="abc123",
            traced_tactics=[
                AnnotatedTactic(
                    tactic="induction x with n ih",
                    annotated_tactic=("induction x with n ih", [
                        Annotation(
                            full_name="induction",
                            def_path="src/tactic/induction.lean",
                            def_pos=Pos(100, 1),
                            def_end_pos=Pos(100, 10)
                        )
                    ]),
                    state_before="⊢ ∀ x y : ℕ, x + y = y + x",
                    state_after="2 goals\ncase zero\n⊢ ∀ y : ℕ, 0 + y = y + 0\ncase succ\nn : ℕ\nih : ∀ y : ℕ, n + y = y + n\n⊢ ∀ y : ℕ, succ n + y = y + succ n"
                )
            ],
            difficulty_rating=0.7
        )

        theorem2 = Theorem(
            full_name="example.quadratic_formula",
            theorem_statement="∀ a b c x : ℝ, a ≠ 0 → (a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a))",
            file_path=Path("src/sorry_example.lean"),
            start=Pos(10, 1),
            end=Pos(12, 10),
            url="https://github.com/example/repo",
            commit="abc123",
            traced_tactics=[],
            difficulty_rating=0.9
        )

        repo.proven_theorems.append(theorem1)
        repo.sorry_theorems_unproved.append(theorem2)

        premise_file = PremiseFile(
            path=Path("src/premise.lean"),
            imports=["import data.real.basic"],
            premises=[
                Premise(
                    full_name="sqrt_squared",
                    code="theorem sqrt_squared (x : ℝ) (h : x ≥ 0) : √(x^2) = x := sorry",
                    start=Pos(1, 1),
                    end=Pos(1, 70),
                    kind="theorem"
                )
            ]
        )

        repo.premise_files.append(premise_file)
        repo.files_traced.append(Path("src/example.lean"))
        return repo
    
    def test_unicode_serialization_deserialization(self):
        json_file = "test_unicode_database.json"
        self.db.to_json(json_file)
        
        deserialized_db = DynamicDatabase.from_json(json_file)
        
        assert len(self.db.repositories) == len(deserialized_db.repositories)
        
        original_repo = self.db.repositories[0]
        deserialized_repo = deserialized_db.repositories[0]
        
        assert original_repo.name == deserialized_repo.name
        assert original_repo.metadata["key"] == deserialized_repo.metadata["key"]
        
        original_theorem1 = original_repo.proven_theorems[0]
        deserialized_theorem1 = deserialized_repo.proven_theorems[0]
        
        assert original_theorem1.theorem_statement == deserialized_theorem1.theorem_statement
        assert original_theorem1.traced_tactics[0].state_before == deserialized_theorem1.traced_tactics[0].state_before
        assert original_theorem1.traced_tactics[0].state_after == deserialized_theorem1.traced_tactics[0].state_after
        
        original_theorem2 = original_repo.sorry_theorems_unproved[0]
        deserialized_theorem2 = deserialized_repo.sorry_theorems_unproved[0]
        
        assert original_theorem2.theorem_statement == deserialized_theorem2.theorem_statement
        
        original_premise = original_repo.premise_files[0].premises[0]
        deserialized_premise = deserialized_repo.premise_files[0].premises[0]
        
        assert original_premise.code == deserialized_premise.code
    
    def test_unicode_modification(self):
        json_file = "test_unicode_database.json"
        self.db.to_json(json_file)
        
        deserialized_db = DynamicDatabase.from_json(json_file)
        
        repo = deserialized_db.get_repository("https://github.com/example/repo", "abc123")
        assert repo is not None
        
        sorry_theorem = repo.sorry_theorems_unproved[0]
        
        sorry_theorem.traced_tactics = [
            AnnotatedTactic(
                tactic="intros a b c x h_a_nonzero",
                annotated_tactic=("intros a b c x h_a_nonzero", []),
                state_before="⊢ ∀ a b c x : ℝ, a ≠ 0 → (a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a))",
                state_after="a b c x : ℝ\nh_a_nonzero : a ≠ 0\n⊢ a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a)"
            ),
            AnnotatedTactic(
                tactic="apply iff.intro",
                annotated_tactic=("apply iff.intro", []),
                state_before="a b c x : ℝ\nh_a_nonzero : a ≠ 0\n⊢ a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a)",
                state_after="2 goals\ncase mp\na b c x : ℝ\nh_a_nonzero : a ≠ 0\n⊢ a * x² + b * x + c = 0 → x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a)\ncase mpr\na b c x : ℝ\nh_a_nonzero : a ≠ 0\n⊢ (x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a)) → a * x² + b * x + c = 0"
            ),
            AnnotatedTactic(
                tactic="sorry",
                annotated_tactic=("sorry", []),
                state_before="2 goals\ncase mp\na b c x : ℝ\nh_a_nonzero : a ≠ 0\n⊢ a * x² + b * x + c = 0 → x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a)\ncase mpr\na b c x : ℝ\nh_a_nonzero : a ≠ 0\n⊢ (x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a)) → a * x² + b * x + c = 0",
                state_after="no goals"
            )
        ]
        
        repo.change_sorry_to_proven(sorry_theorem)
        deserialized_db.update_json(json_file)
        updated_db = DynamicDatabase.from_json(json_file)
        updated_repo = updated_db.get_repository("https://github.com/example/repo", "abc123")
        assert updated_repo is not None
        
        assert len(updated_repo.sorry_theorems_unproved) == 0
        assert len(updated_repo.sorry_theorems_proved) == 1
        
        updated_theorem = updated_repo.sorry_theorems_proved[0]
        assert updated_theorem.full_name == "example.quadratic_formula"
        assert len(updated_theorem.traced_tactics) == 3
        assert "√(b² - 4*a*c)" in updated_theorem.traced_tactics[0].state_before
        assert "↔" in updated_theorem.traced_tactics[1].state_before

class TestDynamicDatabase(unittest.TestCase):
    def setUp(self):
        self.db = DynamicDatabase()
        self.repo = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            date_processed=datetime.datetime.now(),
            metadata={"key": "value"},
        )

    def test_add_repository(self):
        self.db.add_repository(self.repo)
        self.assertEqual(len(self.db.repositories), 1)

    def test_get_repository(self):
        self.db.add_repository(self.repo)
        retrieved_repo = self.db.get_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(retrieved_repo, self.repo)

    def test_update_repository(self):
        self.db.add_repository(self.repo)
        updated_repo = Repository(
            url="https://github.com/test/repo",
            name="Updated Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            date_processed=datetime.datetime.now(),
            metadata={"key": "new_value"},
        )
        self.db.update_repository(updated_repo)
        retrieved_repo = self.db.get_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(retrieved_repo.name, "Updated Test Repo")
        self.assertEqual(retrieved_repo.metadata["key"], "new_value")

    def test_delete_repository(self):
        self.db.add_repository(self.repo)
        self.db.delete_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(len(self.db.repositories), 0)

    def test_to_json_and_from_json(self):
        self.db.add_repository(self.repo)
        json_file = "test_database.json"
        self.db.to_json(json_file)
        loaded_db = DynamicDatabase.from_json(json_file)
        self.assertEqual(len(loaded_db.repositories), 1)
        loaded_repo = loaded_db.get_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(loaded_repo.name, "Test Repo")
        self.assertEqual(loaded_repo.metadata["key"], "value")

class TestDynamicDatabasePFR(unittest.TestCase):
    def setUp(self):
        self.db = DynamicDatabase()
        self.sample_repo = self.create_sample_repo()
        self.db.add_repository(self.sample_repo)

    def create_sample_repo(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        lean_git_repo = LeanGitRepo(url, commit)
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + dir_name + "_updated"
        config = lean_git_repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        theorems_folder = dst_dir + "/random"
        premise_files_corpus = dst_dir + "/corpus.jsonl"
        files_traced = dst_dir + "/traced_files.jsonl"
        pr_url = None
        data = {
            "url": lean_git_repo.url,
            "name": "/".join(lean_git_repo.url.split("/")[-2:]),
            "commit": lean_git_repo.commit,
            "lean_version": v,
            "lean_dojo_version": lean_dojo.__version__,
            "date_processed": datetime.datetime.now(),
            "metadata": {
                "key": "value",
                "unicode": "ユニコード ✨"
            },
            "theorems_folder": theorems_folder,
            "premise_files_corpus": premise_files_corpus,
            "files_traced": files_traced,
            "pr_url": pr_url
        }
        repo = Repository.from_dict(data)
        return repo

    def test_repository_creation(self):
        self.assertIsNotNone(self.sample_repo)
        self.assertEqual(self.sample_repo.url, "https://github.com/teorth/pfr")
        self.assertEqual(self.sample_repo.commit, "6a5082ee465f9e44cea479c7b741b3163162bb7e")

    def test_theorem_loading(self):
        self.assertGreater(len(self.sample_repo.proven_theorems), 0)
        self.assertGreater(len(self.sample_repo.sorry_theorems_unproved), 0)

        theorem = next(t for t in self.sample_repo.proven_theorems if t.full_name == "ContinuousLinearMap.opNorm_lsmul")
        self.assertIsNotNone(theorem)
        self.assertEqual(theorem.file_path, Path(".lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean"))
        self.assertEqual(theorem.start, Pos(281, 1))
        self.assertEqual(theorem.end, Pos(290, 26))

    def test_traced_tactics(self):
        theorem = next(t for t in self.sample_repo.proven_theorems if t.full_name == "ContinuousLinearMap.opNorm_lsmul")
        self.assertGreater(len(theorem.traced_tactics), 0)

        first_tactic = theorem.traced_tactics[0]
        self.assertEqual(first_tactic.tactic, "refine' ContinuousLinearMap.opNorm_eq_of_bounds zero_le_one (fun x => _) fun N _ h => _")
        self.assertIn("ContinuousLinearMap.opNorm_eq_of_bounds", first_tactic.annotated_tactic[0])

    def test_premise_loading(self):
        self.assertGreater(len(self.sample_repo.premise_files), 0)

        premise_file = next(pf for pf in self.sample_repo.premise_files if pf.path == Path(".lake/packages/lean4/src/lean/Init/Prelude.lean"))
        self.assertIsNotNone(premise_file)
        self.assertGreater(len(premise_file.premises), 0)

        premise = next(p for p in premise_file.premises if p.full_name == "id")
        self.assertIsNotNone(premise)
        self.assertIn("def id", premise.code)

    def test_serialization_deserialization(self):
        json_file = "test_pfr_database.json"
        self.db.to_json(json_file)

        deserialized_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(self.db.repositories), len(deserialized_db.repositories))

        original_repo = self.db.repositories[0]
        deserialized_repo = deserialized_db.repositories[0]

        self.assertEqual(original_repo.name, deserialized_repo.name)
        self.assertEqual(original_repo.commit, deserialized_repo.commit)
        self.assertEqual(len(original_repo.proven_theorems), len(deserialized_repo.proven_theorems))
        self.assertEqual(len(original_repo.premise_files), len(deserialized_repo.premise_files))

    def test_generate_dataset_structure(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        self.assertTrue(dst_dir.exists())
        self.assertTrue((dst_dir / "random").exists())
        self.assertTrue((dst_dir / "novel_premises").exists())
        self.assertTrue((dst_dir / "random" / "train.json").exists())
        self.assertTrue((dst_dir / "random" / "val.json").exists())
        self.assertTrue((dst_dir / "random" / "test.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "train.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "val.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "test.json").exists())
        self.assertTrue((dst_dir / "corpus.jsonl").exists())
        self.assertTrue((dst_dir / "traced_files.jsonl").exists())
        self.assertTrue((dst_dir / "metadata.json").exists())

    def test_generated_dataset_content(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        with open(dst_dir / "random" / "train.json", 'r') as f:
            train_data = json.load(f)
            self.assertIsInstance(train_data, list)
            self.assertGreater(len(train_data), 0)
            first_theorem = train_data[0]
            self.assertIn("url", first_theorem)
            self.assertIn("commit", first_theorem)
            self.assertIn("file_path", first_theorem)
            self.assertIn("full_name", first_theorem)
            self.assertIn("theorem_statement", first_theorem)
            self.assertIn("start", first_theorem)
            self.assertIn("end", first_theorem)
            self.assertIn("traced_tactics", first_theorem)

        with open(dst_dir / "corpus.jsonl", 'r') as f:
            first_line = f.readline().strip()
            first_premise_file = json.loads(first_line)
            self.assertIn("path", first_premise_file)
            self.assertIn("imports", first_premise_file)
            self.assertIn("premises", first_premise_file)

        with open(dst_dir / "traced_files.jsonl", 'r') as f:
            first_line = f.readline().strip()
            first_traced_file = json.loads(first_line)
            self.assertIn("traced_file_path", first_traced_file)

        with open(dst_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.assertIn("repositories", metadata)
            self.assertIn("total_theorems", metadata)
            self.assertIn("num_proven_theorems", metadata)
            self.assertIn("num_sorry_theorems", metadata)
            self.assertIn("num_premise_files", metadata)
            self.assertIn("num_premises", metadata)
            self.assertIn("num_files_traced", metadata)

    def test_dataset_splitting(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        for strategy in ['random', 'novel_premises']:
            train_set = set()
            val_set = set()
            test_set = set()

            with open(dst_dir / strategy / "train.json", 'r') as f:
                train_data = json.load(f)
                train_set = set((item['full_name'], item['file_path'], tuple(item['start']), tuple(item['end'])) for item in train_data)

            with open(dst_dir / strategy / "val.json", 'r') as f:
                val_data = json.load(f)
                val_set = set((item['full_name'], item['file_path'], tuple(item['start']), tuple(item['end'])) for item in val_data)

            with open(dst_dir / strategy / "test.json", 'r') as f:
                test_data = json.load(f)
                test_set = set((item['full_name'], item['file_path'], tuple(item['start']), tuple(item['end'])) for item in test_data)

            self.assertGreater(len(train_set), 0)
            self.assertGreater(len(val_set), 0)
            self.assertGreater(len(test_set), 0)

            self.assertEqual(len(train_set.intersection(val_set)), 0)
            self.assertEqual(len(train_set.intersection(test_set)), 0)
            self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_dataset_consistency(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        # Check that all theorems in the dataset are from the original repository
        all_theorems = set(thm.full_name for thm in self.sample_repo.get_all_theorems)

        for strategy in ['random', 'novel_premises']:
            for split in ['train', 'val', 'test']:
                with open(dst_dir / strategy / f"{split}.json", 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.assertIn(item['full_name'], all_theorems)

    def test_compare_manual_and_dynamic_datasets(self):
        random.seed(3407)

        manual_dataset_path = Path(RAID_DIR) / DATA_DIR / "pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_updated"
        dynamic_dataset_path = Path(RAID_DIR) / DATA_DIR / "pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_generated"

        self.db.generate_merged_dataset(dynamic_dataset_path)
        
        for strategy in ['random', 'novel_premises']:
            logger.info(f"Comparing datasets for {strategy} strategy")
            manual_theorems = []
            dynamic_theorems = []

            for split in ['train', 'val', 'test']:
                logger.info(f"Loading {split} split for {strategy} strategy")
                manual_file = manual_dataset_path / strategy / f"{split}.json"
                dynamic_file = dynamic_dataset_path / strategy / f"{split}.json"
                
                with open(manual_file, 'r') as f:
                    manual_data = json.load(f)
                    manual_theorems.extend(manual_data)
                logger.info(f"Loaded {len(manual_data)} theorems from manual {split} split")
                
                with open(dynamic_file, 'r') as f:
                    dynamic_data = json.load(f)
                    dynamic_theorems.extend(dynamic_data)
                logger.info(f"Loaded {len(dynamic_data)} theorems from dynamic {split} split")
            
            assert len(manual_theorems) == len(dynamic_theorems), "Manual and dynamic datasets have different number of theorems"
            logger.info(f"Comparing {len(manual_theorems)} manual theorems with {len(dynamic_theorems)} dynamic theorems for {strategy} strategy")
            self.assertTrue(self._fast_compare_theorems(manual_theorems, dynamic_theorems), 
                        f"Theorem content for {strategy} strategy does not match")
            logger.info(f"Theorem content for {strategy} strategy matches")

        self.maxDiff = None
        logger.info("Comparing corpus and traced files")
        with open(manual_dataset_path / "corpus.jsonl", 'r') as f:
            manual_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_corpus)} items from manual corpus")

        with open(dynamic_dataset_path / "corpus.jsonl", 'r') as f:
            dynamic_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_corpus)} items from dynamic corpus")

        assert len(manual_corpus) == len(dynamic_corpus), "Manual and dynamic datasets have different number of premise files"
        logger.info("Comparing corpus content")
        try:
            self.assertCountEqual(manual_corpus, dynamic_corpus)
            logger.info("Corpus content matches")
        except AssertionError as e:
            logger.info("Corpus content mismatch:")
            logger.info(str(e))
            raise

        with open(manual_dataset_path / "traced_files.jsonl", 'r') as f:
            manual_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_traced)} items from manual traced files")

        with open(dynamic_dataset_path / "traced_files.jsonl", 'r') as f:
            dynamic_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_traced)} items from dynamic traced files")

        assert len(manual_traced) == len(dynamic_traced), "Manual and dynamic datasets have different number of traced files"
        logger.info("Comparing traced files content")
        try:
            self.assertCountEqual(manual_traced, dynamic_traced)
            logger.info("Traced files content matches")
        except AssertionError as e:
            logger.info("Traced files content mismatch:")
            logger.info(str(e))
            raise

    def _fast_compare_theorems(self, manual_theorems, dynamic_theorems):
        logger.info(f"Converting {len(manual_theorems)} manual theorems to hashable format")
        manual_set = set(map(self._theorem_to_hashable, manual_theorems))
        assert len(manual_set) == len(manual_theorems), "Manual theorems contain duplicates"
        logger.info(f"Converting {len(dynamic_theorems)} dynamic theorems to hashable format")
        dynamic_set = set(map(self._theorem_to_hashable, dynamic_theorems))
        assert len(dynamic_set) == len(dynamic_theorems), "Dynamic theorems contain duplicates"

        logger.info("Comparing theorem sets")
        only_in_manual = manual_set - dynamic_set
        only_in_dynamic = dynamic_set - manual_set

        if only_in_manual or only_in_dynamic:
            # Sort by file path and full name
            only_in_manual = sorted(only_in_manual, key=lambda x: (x[0], x[1]))
            only_in_dynamic = sorted(only_in_dynamic, key=lambda x: (x[0], x[1]))
            if only_in_manual:
                logger.info(f"{len(only_in_manual)} theorems only in manual dataset")
                for i, thm in enumerate(only_in_manual):
                    if i >= 10:
                        break
                    logger.info(f"Manual only: {thm[1]} in {thm[0]}")
                    logger.info(f"  Start: {thm[2]}, End: {thm[3]}")
            if only_in_dynamic:
                logger.info(f"{len(only_in_dynamic)} theorems only in dynamic dataset")
                for i, thm in enumerate(only_in_dynamic):
                    if i >= 10:
                        break
                    logger.info(f"Dynamic only: {thm[1]} in {thm[0]}")
                    logger.info(f"  Start: {thm[2]}, End: {thm[3]}")
            return False
        return True

    def _theorem_to_hashable(self, theorem):
        return (
            theorem['file_path'],
            theorem['full_name'],
            tuple(theorem['start']),
            tuple(theorem['end']),
        )

    def _tactic_to_hashable(self, tactic):
        return (
            tactic['tactic'],
            tactic['annotated_tactic'][0],
            tuple((a['full_name'], a['def_path'], tuple(a['def_pos']), tuple(a['def_end_pos']))
                for a in tactic['annotated_tactic'][1]),
            tactic['state_before'],
            tactic['state_after']
        )

    def test_unicode_handling_in_dataset(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        with open(dst_dir / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.assertIn('repositories', metadata, "No 'repositories' key in metadata")
            self.assertGreater(len(metadata['repositories']), 0, "No repositories in metadata")
            repo = metadata['repositories'][0]
            self.assertIn('metadata', repo, "No 'metadata' key in repository")
            repo_metadata = repo['metadata']
            self.assertIn('unicode', repo_metadata, "No 'unicode' key in repository metadata")
            self.assertIn("ユニコード", repo_metadata['unicode'], "Unicode string not found in metadata")
            self.assertIn("ユニコード", metadata['repositories'][0]['metadata']['unicode'])

    def tearDown(self):
        # Clean up generated files after tests
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)

    def test_theorem_statement(self):
        theorem = next(t for t in self.sample_repo.proven_theorems if t.full_name == "ContinuousLinearMap.opNorm_lsmul")
        self.assertIsNotNone(theorem.theorem_statement)
        self.assertIn("opNorm_lsmul", theorem.theorem_statement)

    def test_unicode_handling(self):
        self.assertIsNotNone(self.sample_repo.metadata)
        self.assertIsNotNone(self.sample_repo.metadata["unicode"])
        self.assertIn("ユニコード", self.sample_repo.metadata["unicode"])

    def test_difficulty_rating(self):
        for theorem in self.sample_repo.proven_theorems:
            if theorem.difficulty_rating is not None:
                self.assertGreaterEqual(theorem.difficulty_rating, 0.0)
                self.assertLessEqual(theorem.difficulty_rating, 1.0)

    def test_file_tracing(self):
        self.assertGreater(len(self.sample_repo.files_traced), 0)
        self.assertIn(Path("PFR/Mathlib/GroupTheory/Torsion.lean"), self.sample_repo.files_traced)
        self.assertIn(Path(".lake/packages/batteries/Batteries/Data/List/Lemmas.lean"), self.sample_repo.files_traced)

class TestDynamicDatabasePFRNewVersion(unittest.TestCase):
    def setUp(self):
        self.db = DynamicDatabase()
        self.sample_repo_PFR = self.create_sample_repo_PFR()
        self.sample_repo_new_version = self.create_sample_repo_new_version()
        self.db.add_repository(self.sample_repo_PFR)
        self.db.add_repository(self.sample_repo_new_version)

    def create_sample_repo_PFR(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        lean_git_repo = LeanGitRepo(url, commit)
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + dir_name + "_updated"
        config = lean_git_repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        theorems_folder = dst_dir + "/random"
        premise_files_corpus = dst_dir + "/corpus.jsonl"
        files_traced = dst_dir + "/traced_files.jsonl"
        pr_url = None
        data = {
            "url": lean_git_repo.url,
            "name": "/".join(lean_git_repo.url.split("/")[-2:]),
            "commit": lean_git_repo.commit,
            "lean_version": v,
            "lean_dojo_version": lean_dojo.__version__,
            "date_processed": datetime.datetime.now(),
            "metadata": {
                "key": "value",
                "unicode": "ユニコード ✨"
            },
            "theorems_folder": theorems_folder,
            "premise_files_corpus": premise_files_corpus,
            "files_traced": files_traced,
            "pr_url": pr_url
        }
        repo = Repository.from_dict(data)
        return repo
    
    def create_sample_repo_new_version(self):
        url = "https://github.com/Adarsh321123/new-version-test"
        commit = "f465306be03ced999caa157a85558a6c41b3e3f5"
        lean_git_repo = LeanGitRepo(url, commit)
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + dir_name + "_updated"
        config = lean_git_repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        theorems_folder = dst_dir + "/random"
        premise_files_corpus = dst_dir + "/corpus.jsonl"
        files_traced = dst_dir + "/traced_files.jsonl"
        pr_url = None
        data = {
            "url": lean_git_repo.url,
            "name": "/".join(lean_git_repo.url.split("/")[-2:]),
            "commit": lean_git_repo.commit,
            "lean_version": v,
            "lean_dojo_version": lean_dojo.__version__,
            "date_processed": datetime.datetime.now(),
            "metadata": {
                "key": "value",
                "unicode": "ユニコード ✨"
            },
            "theorems_folder": theorems_folder,
            "premise_files_corpus": premise_files_corpus,
            "files_traced": files_traced,
            "pr_url": pr_url
        }
        repo = Repository.from_dict(data)
        return repo

    def test_repository_creation(self):
        self.assertIsNotNone(self.sample_repo_PFR)
        self.assertEqual(self.sample_repo_PFR.url, "https://github.com/teorth/pfr")
        self.assertEqual(self.sample_repo_PFR.commit, "6a5082ee465f9e44cea479c7b741b3163162bb7e")
        self.assertIsNotNone(self.sample_repo_new_version)
        self.assertEqual(self.sample_repo_new_version.url, "https://github.com/Adarsh321123/new-version-test")
        self.assertEqual(self.sample_repo_new_version.commit, "f465306be03ced999caa157a85558a6c41b3e3f5")

    def test_theorem_loading(self):
        self.assertGreater(len(self.sample_repo_PFR.proven_theorems), 0)
        self.assertGreater(len(self.sample_repo_PFR.sorry_theorems_unproved), 0)

        theorem = next(t for t in self.sample_repo_PFR.proven_theorems if t.full_name == "ContinuousLinearMap.opNorm_lsmul")
        self.assertIsNotNone(theorem)
        self.assertEqual(theorem.file_path, Path(".lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean"))
        self.assertEqual(theorem.start, Pos(281, 1))
        self.assertEqual(theorem.end, Pos(290, 26))

        self.assertGreater(len(self.sample_repo_new_version.proven_theorems), 0)
        self.assertGreater(len(self.sample_repo_new_version.sorry_theorems_unproved), 0)

        theorem = next(t for t in self.sample_repo_new_version.proven_theorems if t.full_name == "Ordinal.le_mul_right")
        self.assertIsNotNone(theorem)
        self.assertEqual(theorem.file_path, Path(".lake/packages/mathlib/Mathlib/SetTheory/Ordinal/Arithmetic.lean"))
        self.assertEqual(theorem.start, Pos(742, 1))
        self.assertEqual(theorem.end, Pos(744, 17))

    def test_traced_tactics(self):
        theorem = next(t for t in self.sample_repo_PFR.proven_theorems if t.full_name == "ContinuousLinearMap.opNorm_lsmul")
        self.assertGreater(len(theorem.traced_tactics), 0)

        first_tactic = theorem.traced_tactics[0]
        self.assertEqual(first_tactic.tactic, "refine' ContinuousLinearMap.opNorm_eq_of_bounds zero_le_one (fun x => _) fun N _ h => _")
        self.assertIn("ContinuousLinearMap.opNorm_eq_of_bounds", first_tactic.annotated_tactic[0])

        theorem = next(t for t in self.sample_repo_new_version.proven_theorems if t.full_name == "Ordinal.le_mul_right")
        self.assertGreater(len(theorem.traced_tactics), 0)

        first_tactic = theorem.traced_tactics[0]
        self.assertEqual(first_tactic.tactic, "convert mul_le_mul_right' (one_le_iff_pos.2 hb) a")
        self.assertIn("mul_le_mul_right'", first_tactic.annotated_tactic[0])

    def test_premise_loading(self):
        self.assertGreater(len(self.sample_repo_PFR.premise_files), 0)

        premise_file = next(pf for pf in self.sample_repo_PFR.premise_files if pf.path == Path(".lake/packages/lean4/src/lean/Init/Prelude.lean"))
        self.assertIsNotNone(premise_file)
        self.assertGreater(len(premise_file.premises), 0)

        premise = next(p for p in premise_file.premises if p.full_name == "id")
        self.assertIsNotNone(premise)
        self.assertIn("def id", premise.code)

        self.assertGreater(len(self.sample_repo_new_version.premise_files), 0)

        premise_file = next(pf for pf in self.sample_repo_new_version.premise_files if pf.path == Path(".lake/packages/lean4/src/lean/Init/Prelude.lean"))
        self.assertIsNotNone(premise_file)
        self.assertGreater(len(premise_file.premises), 0)

        premise = next(p for p in premise_file.premises if p.full_name == "id")
        self.assertIsNotNone(premise)
        self.assertIn("def id", premise.code)

    def test_serialization_deserialization(self):
        json_file = "test_pfr_new_version_database.json"
        self.db.to_json(json_file)

        deserialized_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(self.db.repositories), len(deserialized_db.repositories))

        original_repo_PFR = self.db.repositories[0]
        deserialized_repo_PFR = deserialized_db.repositories[0]

        self.assertEqual(original_repo_PFR.name, deserialized_repo_PFR.name)
        self.assertEqual(original_repo_PFR.commit, deserialized_repo_PFR.commit)
        self.assertEqual(len(original_repo_PFR.proven_theorems), len(deserialized_repo_PFR.proven_theorems))
        self.assertEqual(len(original_repo_PFR.premise_files), len(deserialized_repo_PFR.premise_files))

        original_repo_new_version = self.db.repositories[0]
        deserialized_repo_new_version = deserialized_db.repositories[0]

        self.assertEqual(original_repo_new_version.name, deserialized_repo_new_version.name)
        self.assertEqual(original_repo_new_version.commit, deserialized_repo_new_version.commit)
        self.assertEqual(len(original_repo_new_version.proven_theorems), len(deserialized_repo_new_version.proven_theorems))
        self.assertEqual(len(original_repo_new_version.premise_files), len(deserialized_repo_new_version.premise_files))

    def test_generate_dataset_structure(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        self.db.generate_merged_dataset(dst_dir)

        self.assertTrue(dst_dir.exists())
        self.assertTrue((dst_dir / "random").exists())
        self.assertTrue((dst_dir / "novel_premises").exists())
        self.assertTrue((dst_dir / "random" / "train.json").exists())
        self.assertTrue((dst_dir / "random" / "val.json").exists())
        self.assertTrue((dst_dir / "random" / "test.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "train.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "val.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "test.json").exists())
        self.assertTrue((dst_dir / "corpus.jsonl").exists())
        self.assertTrue((dst_dir / "traced_files.jsonl").exists())
        self.assertTrue((dst_dir / "metadata.json").exists())

    def test_generated_dataset_content(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        self.db.generate_merged_dataset(dst_dir)

        with open(dst_dir / "random" / "train.json", 'r') as f:
            train_data = json.load(f)
            self.assertIsInstance(train_data, list)
            self.assertGreater(len(train_data), 0)
            first_theorem = train_data[0]
            self.assertIn("url", first_theorem)
            self.assertIn("commit", first_theorem)
            self.assertIn("file_path", first_theorem)
            self.assertIn("full_name", first_theorem)
            self.assertIn("theorem_statement", first_theorem)
            self.assertIn("start", first_theorem)
            self.assertIn("end", first_theorem)
            self.assertIn("traced_tactics", first_theorem)

        with open(dst_dir / "corpus.jsonl", 'r') as f:
            first_line = f.readline().strip()
            first_premise_file = json.loads(first_line)
            self.assertIn("path", first_premise_file)
            self.assertIn("imports", first_premise_file)
            self.assertIn("premises", first_premise_file)

        with open(dst_dir / "traced_files.jsonl", 'r') as f:
            first_line = f.readline().strip()
            first_traced_file = json.loads(first_line)
            self.assertIn("traced_file_path", first_traced_file)

        with open(dst_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.assertIn("repositories", metadata)
            self.assertIn("total_theorems", metadata)
            self.assertIn("num_proven_theorems", metadata)
            self.assertIn("num_sorry_theorems", metadata)
            self.assertIn("num_premise_files", metadata)
            self.assertIn("num_premises", metadata)
            self.assertIn("num_files_traced", metadata)

    def test_dataset_splitting(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        self.db.generate_merged_dataset(dst_dir)

        for strategy in ['random', 'novel_premises']:
            train_set = set()
            val_set = set()
            test_set = set()

            with open(dst_dir / strategy / "train.json", 'r') as f:
                train_data = json.load(f)
                train_set = set((item['full_name'], item['file_path'], tuple(item['start']), tuple(item['end'])) for item in train_data)

            with open(dst_dir / strategy / "val.json", 'r') as f:
                val_data = json.load(f)
                val_set = set((item['full_name'], item['file_path'], tuple(item['start']), tuple(item['end'])) for item in val_data)

            with open(dst_dir / strategy / "test.json", 'r') as f:
                test_data = json.load(f)
                test_set = set((item['full_name'], item['file_path'], tuple(item['start']), tuple(item['end'])) for item in test_data)

            self.assertGreater(len(train_set), 0)
            self.assertGreater(len(val_set), 0)
            self.assertGreater(len(test_set), 0)

            self.assertEqual(len(train_set.intersection(val_set)), 0)
            self.assertEqual(len(train_set.intersection(test_set)), 0)
            self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_dataset_consistency(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"merged_{dir_name_PFR}_{dir_name_new_version}_generated"
        self.db.generate_merged_dataset(dst_dir)

        # Check that all theorems in the dataset are from the original repositories
        all_theorems_PFR = set(thm.full_name for thm in self.sample_repo_PFR.get_all_theorems)
        all_theorems_new_version = set(thm.full_name for thm in self.sample_repo_new_version.get_all_theorems)

        for strategy in ['random', 'novel_premises']:
            for split in ['train', 'val', 'test']:
                with open(dst_dir / strategy / f"{split}.json", 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.assertIn(item['full_name'], all_theorems_PFR | all_theorems_new_version)
    
    def test_compare_manual_and_dynamic_datasets(self):
        random.seed(3407)

        manual_dataset_path = Path(RAID_DIR) / MERGED_DATA_DIR / "merged_pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_new-version-test_f465306be03ced999caa157a85558a6c41b3e3f5_updated"
        dynamic_dataset_path = Path(RAID_DIR) / MERGED_DATA_DIR / "merged_pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_new-version-test_f465306be03ced999caa157a85558a6c41b3e3f5_generated"

        self.db.generate_merged_dataset(dynamic_dataset_path)
        
        for strategy in ['random', 'novel_premises']:
            logger.info(f"Comparing datasets for {strategy} strategy")
            manual_theorems = []
            dynamic_theorems = []

            for split in ['train', 'val', 'test']:
                logger.info(f"Loading {split} split for {strategy} strategy")
                manual_file = manual_dataset_path / strategy / f"{split}.json"
                dynamic_file = dynamic_dataset_path / strategy / f"{split}.json"
                
                with open(manual_file, 'r') as f:
                    manual_data = json.load(f)
                    manual_theorems.extend(manual_data)
                logger.info(f"Loaded {len(manual_data)} theorems from manual {split} split")
                
                with open(dynamic_file, 'r') as f:
                    dynamic_data = json.load(f)
                    dynamic_theorems.extend(dynamic_data)
                logger.info(f"Loaded {len(dynamic_data)} theorems from dynamic {split} split")
            
            logger.info(f"Comparing {len(manual_theorems)} manual theorems with {len(dynamic_theorems)} dynamic theorems for {strategy} strategy")

            # The manual code has a bug where it allows duplicate theorems as long as they exist in different repositories.
            # As such, we need to remove these duplicates.
            manual_dict = {}
            manual_duplicates = []
            for thm in manual_theorems:
                key = self._theorem_to_key(thm)
                if key in manual_dict:
                    manual_duplicates.append((key, thm, manual_dict[key]))
                else:
                    manual_dict[key] = thm
            deduplicated_manual_theorems = list(manual_dict.values())
            
            dynamic_dict = {self._theorem_to_key(t): t for t in dynamic_theorems}

            logger.info(f"After deduplication - Manual theorems: {len(deduplicated_manual_theorems)}, Dynamic theorems: {len(dynamic_theorems)}")
            
            only_in_manual = set(manual_dict.keys()) - set(dynamic_dict.keys())
            only_in_dynamic = set(dynamic_dict.keys()) - set(manual_dict.keys())
            
            if only_in_manual:
                logger.error(f"{len(only_in_manual)} theorems only in manual dataset for {strategy}")
                for key in list(only_in_manual)[:1]:
                    manual_thm = manual_dict[key]
                    logger.error(f"Manual only: {manual_thm['full_name']} in {manual_thm['file_path']}")
                    logger.error(f"  URL: {manual_thm['url']}, Commit: {manual_thm['commit']}")
                    logger.error(f"  Start: {manual_thm['start']}, End: {manual_thm['end']}")
                    logger.error(f"  Theorem statement: {manual_thm['theorem_statement'][:100]}...")  # First 100 chars
            
            if only_in_dynamic:
                logger.error(f"{len(only_in_dynamic)} theorems only in dynamic dataset for {strategy}")
                for key in list(only_in_dynamic)[:1]:
                    dynamic_thm = dynamic_dict[key]
                    logger.error(f"Dynamic only: {dynamic_thm['full_name']} in {dynamic_thm['file_path']}")
                    logger.error(f"  URL: {dynamic_thm['url']}, Commit: {dynamic_thm['commit']}")
                    logger.error(f"  Start: {dynamic_thm['start']}, End: {dynamic_thm['end']}")
                    logger.error(f"  Theorem statement: {dynamic_thm['theorem_statement'][:100]}...")  # First 100 chars
            
            self.assertEqual(len(only_in_manual), 0, f"Theorems found only in manual dataset for {strategy}")
            self.assertEqual(len(only_in_dynamic), 0, f"Theorems found only in dynamic dataset for {strategy}")
            
            assert len(set(manual_dict.keys())) == len(set(dynamic_dict.keys())), "Manual and dynamic datasets have different number of theorems"
            self.assertTrue(self._fast_compare_theorems(deduplicated_manual_theorems, dynamic_theorems), 
                        f"Theorem content for {strategy} strategy does not match")
            logger.info(f"Theorem content for {strategy} strategy matches after deduplication")

        self.maxDiff = None
        logger.info("Comparing corpus and traced files")
        with open(manual_dataset_path / "corpus.jsonl", 'r') as f:
            manual_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_corpus)} items from manual corpus")

        with open(dynamic_dataset_path / "corpus.jsonl", 'r') as f:
            dynamic_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_corpus)} items from dynamic corpus")

        manual_corpus_dict = {item['path']: item for item in manual_corpus}
        deduplicated_manual_corpus = list(manual_corpus_dict.values())
        dynamic_corpus_dict = {item['path']: item for item in dynamic_corpus}

        logger.info(f"Manual corpus: {len(manual_corpus)} items, {len(deduplicated_manual_corpus)} unique")
        logger.info(f"Dynamic corpus: {len(dynamic_corpus)} items")

        only_in_manual_corpus = set(manual_corpus_dict.keys()) - set(dynamic_corpus_dict.keys())
        only_in_dynamic_corpus = set(dynamic_corpus_dict.keys()) - set(manual_corpus_dict.keys())

        self.assertEqual(len(only_in_manual_corpus), 0, "Corpus items found only in manual dataset")
        self.assertEqual(len(only_in_dynamic_corpus), 0, "Corpus items found only in dynamic dataset")

        assert len(set(dynamic_corpus_dict.keys())) == len(set(dynamic_corpus_dict.keys())), "Manual and dynamic datasets have different number of premise files"
        logger.info("Comparing corpus content")
        try:
            self.assertCountEqual(deduplicated_manual_corpus, dynamic_corpus)
            logger.info("Corpus content matches after deduplication")
        except AssertionError as e:
            logger.info("Corpus content mismatch:")
            logger.info(str(e))
            raise

        with open(manual_dataset_path / "traced_files.jsonl", 'r') as f:
            manual_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_traced)} items from manual traced files")

        with open(dynamic_dataset_path / "traced_files.jsonl", 'r') as f:
            dynamic_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_traced)} items from dynamic traced files")

        manual_traced_dict = {item['traced_file_path']: item for item in manual_traced}
        deduplicated_manual_traced = list(manual_traced_dict.values())
        logger.info(f"Manual traced files: {len(manual_traced)} items, {len(deduplicated_manual_traced)} unique")
        logger.info(f"Dynamic traced files: {len(dynamic_traced)} items")

        logger.info("Comparing traced files content")
        try:
            self.assertCountEqual(deduplicated_manual_traced, dynamic_traced)
            logger.info("Traced files content matches after deduplication")
        except AssertionError as e:
            logger.info("Traced files content mismatch:")
            logger.info(str(e))
            raise

    def _theorem_to_key(self, theorem):
        return (
            theorem['file_path'],
            theorem['full_name'],
            tuple(theorem['start']),
            tuple(theorem['end'])
        )

    def _fast_compare_theorems(self, manual_theorems, dynamic_theorems):
        logger.info(f"Converting {len(manual_theorems)} manual theorems to hashable format")
        manual_set = set(map(self._theorem_to_hashable, manual_theorems))
        assert len(manual_set) == len(manual_theorems), "Manual theorems contain duplicates"
        logger.info(f"Converting {len(dynamic_theorems)} dynamic theorems to hashable format")
        dynamic_set = set(map(self._theorem_to_hashable, dynamic_theorems))
        assert len(dynamic_set) == len(dynamic_theorems), "Dynamic theorems contain duplicates"

        logger.info("Comparing theorem sets")
        only_in_manual = manual_set - dynamic_set
        only_in_dynamic = dynamic_set - manual_set

        if only_in_manual or only_in_dynamic:
            # Sort by file path and full name
            only_in_manual = sorted(only_in_manual, key=lambda x: (x[0], x[1]))
            only_in_dynamic = sorted(only_in_dynamic, key=lambda x: (x[0], x[1]))
            if only_in_manual:
                logger.info(f"{len(only_in_manual)} theorems only in manual dataset")
                for i, thm in enumerate(only_in_manual):
                    if i >= 10:
                        break
                    logger.info(f"Manual only: {thm[1]} in {thm[0]}")
                    logger.info(f"  Start: {thm[2]}, End: {thm[3]}")
            if only_in_dynamic:
                logger.info(f"{len(only_in_dynamic)} theorems only in dynamic dataset")
                for i, thm in enumerate(only_in_dynamic):
                    if i >= 10:
                        break
                    logger.info(f"Dynamic only: {thm[1]} in {thm[0]}")
                    logger.info(f"  Start: {thm[2]}, End: {thm[3]}")
            return False
        return True

    def _theorem_to_hashable(self, theorem):
        return (
            theorem['file_path'],
            theorem['full_name'],
            tuple(theorem['start']),
            tuple(theorem['end']),
        )

    def _tactic_to_hashable(self, tactic):
        return (
            tactic['tactic'],
            tactic['annotated_tactic'][0],
            tuple((a['full_name'], a['def_path'], tuple(a['def_pos']), tuple(a['def_end_pos']))
                for a in tactic['annotated_tactic'][1]),
            tactic['state_before'],
            tactic['state_after']
        )
    
    def test_unicode_handling_in_dataset(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        self.db.generate_merged_dataset(dst_dir)

        with open(dst_dir / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.assertIn('repositories', metadata, "No 'repositories' key in metadata")
            self.assertGreater(len(metadata['repositories']), 0, "No repositories in metadata")
            repo_PFR = metadata['repositories'][0]
            self.assertIn('metadata', repo_PFR, "No 'metadata' key in repository")
            repo_PFR_metadata = repo_PFR['metadata']
            self.assertIn('unicode', repo_PFR_metadata, "No 'unicode' key in repository metadata")
            self.assertIn("ユニコード", repo_PFR_metadata['unicode'], "Unicode string not found in metadata")
            self.assertIn("ユニコード", metadata['repositories'][0]['metadata']['unicode'])

            self.assertGreater(len(metadata['repositories']), 1, "Only one repository in metadata")
            repo_new_version = metadata['repositories'][1]
            self.assertIn('metadata', repo_new_version, "No 'metadata' key in repository")
            repo_new_version_metadata = repo_new_version['metadata']
            self.assertIn('unicode', repo_new_version_metadata, "No 'unicode' key in repository metadata")
            self.assertIn("ユニコード", repo_new_version_metadata['unicode'], "Unicode string not found in metadata")
            self.assertIn("ユニコード", metadata['repositories'][1]['metadata']['unicode'])

    def tearDown(self):
        # Clean up generated files after tests
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"merged_{dir_name_PFR}_{dir_name_new_version}_generated"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)

    def test_theorem_statement(self):
        theorem = next(t for t in self.sample_repo_PFR.proven_theorems if t.full_name == "ContinuousLinearMap.opNorm_lsmul")
        self.assertIsNotNone(theorem.theorem_statement)
        self.assertIn("opNorm_lsmul", theorem.theorem_statement)

        theorem = next(t for t in self.sample_repo_new_version.proven_theorems if t.full_name == "Ordinal.le_mul_right")
        self.assertIsNotNone(theorem.theorem_statement)
        self.assertIn("le_mul_right", theorem.theorem_statement)

    def test_unicode_handling(self):
        self.assertIsNotNone(self.sample_repo_PFR.metadata)
        self.assertIsNotNone(self.sample_repo_PFR.metadata["unicode"])
        self.assertIn("ユニコード", self.sample_repo_PFR.metadata["unicode"])

        self.assertIsNotNone(self.sample_repo_new_version.metadata)
        self.assertIsNotNone(self.sample_repo_new_version.metadata["unicode"])
        self.assertIn("ユニコード", self.sample_repo_new_version.metadata["unicode"])

    def test_difficulty_rating(self):
        for theorem in self.sample_repo_PFR.proven_theorems:
            if theorem.difficulty_rating is not None:
                self.assertGreaterEqual(theorem.difficulty_rating, 0.0)
                self.assertLessEqual(theorem.difficulty_rating, 1.0)

        for theorem in self.sample_repo_new_version.proven_theorems:
            if theorem.difficulty_rating is not None:
                self.assertGreaterEqual(theorem.difficulty_rating, 0.0)
                self.assertLessEqual(theorem.difficulty_rating, 1.0)

    def test_file_tracing(self):
        self.assertGreater(len(self.sample_repo_PFR.files_traced), 0)
        self.assertIn(Path("PFR/Mathlib/GroupTheory/Torsion.lean"), self.sample_repo_PFR.files_traced)
        self.assertIn(Path(".lake/packages/batteries/Batteries/Data/List/Lemmas.lean"), self.sample_repo_PFR.files_traced)

        self.assertGreater(len(self.sample_repo_new_version.files_traced), 0)
        self.assertIn(Path("NewVersionTest/ExercisesOne.lean"), self.sample_repo_new_version.files_traced)
        self.assertIn(Path(".lake/packages/batteries/Batteries/Data/List/Lemmas.lean"), self.sample_repo_new_version.files_traced)

class TestDynamicDatabaseProver(unittest.TestCase):
    def setUp(self):
        self.db = DynamicDatabase()
        self.repo = Repository(
            url="https://github.com/test/repo",
            name="test_repo",
            commit="abcdef1234567890",
            lean_version="4.0.0",
            lean_dojo_version="1.0.0",
            date_processed=datetime.datetime.now(),
            metadata={"key": "value"},
            sorry_theorems_unproved=[
                Theorem(
                    full_name="test_theorem",
                    file_path=Path("src/test.lean"),
                    start=Pos(1, 1),
                    end=Pos(10, 1),
                    url="https://github.com/test/repo",
                    commit="abcdef1234567890",
                    theorem_statement="theorem test_theorem : 2 + 2 = 4 := sorry"
                )
            ]
        )
        self.db.add_repository(self.repo)

    def test_create_annotated_tactic(self):
        tactic = "rw [add_comm]"
        annotated_tactic = AnnotatedTactic(
            tactic=tactic,
            annotated_tactic=(tactic, []),
            state_before="",
            state_after=""
        )
        self.assertEqual(annotated_tactic.tactic, tactic)
        self.assertEqual(annotated_tactic.annotated_tactic, (tactic, []))
        self.assertEqual(annotated_tactic.state_before, "")
        self.assertEqual(annotated_tactic.state_after, "")

    def test_update_theorem_with_proof(self):
        theorem = self.repo.sorry_theorems_unproved[0]
        traced_tactics = [
            AnnotatedTactic(
                tactic="rw [add_comm]",
                annotated_tactic=("rw [add_comm]", []),
                state_before="⊢ 2 + 2 = 4",
                state_after="⊢ 2 + 2 = 4"
            ),
            AnnotatedTactic(
                tactic="refl",
                annotated_tactic=("refl", []),
                state_before="⊢ 2 + 2 = 4",
                state_after="no goals"
            )
        ]
        theorem.traced_tactics = traced_tactics
        self.repo.change_sorry_to_proven(theorem)
        self.db.update_repository(self.repo)

        updated_repo = self.db.get_repository(self.repo.url, self.repo.commit)
        self.assertEqual(len(updated_repo.sorry_theorems_proved), 1)
        self.assertEqual(len(updated_repo.sorry_theorems_unproved), 0)
        self.assertEqual(updated_repo.sorry_theorems_proved[0].traced_tactics, traced_tactics)

    def test_update_theorem_with_proof_and_json(self):
        json_file = "temp_file.json"
        theorem = self.repo.sorry_theorems_unproved[0]
        
        traced_tactics = [
            AnnotatedTactic(
                tactic="rw [add_comm]",
                annotated_tactic=("rw [add_comm]", []),
                state_before="",
                state_after=""
            )
        ]
        
        theorem.traced_tactics = traced_tactics
        self.repo.change_sorry_to_proven(theorem)
        self.db.update_repository(self.repo)
        self.db.to_json(json_file)

        loaded_db = DynamicDatabase.from_json(json_file)
        loaded_repo = loaded_db.get_repository(self.repo.url, self.repo.commit)

        self.assertEqual(len(loaded_repo.sorry_theorems_proved), 1)
        self.assertEqual(len(loaded_repo.sorry_theorems_unproved), 0)
        proved_theorem = loaded_repo.sorry_theorems_proved[0]
        self.assertEqual(proved_theorem.full_name, theorem.full_name)
        self.assertEqual(proved_theorem.file_path, theorem.file_path)
        self.assertEqual(proved_theorem.start, theorem.start)
        self.assertEqual(proved_theorem.end, theorem.end)
        
        self.assertEqual(len(proved_theorem.traced_tactics), 1)
        loaded_tactic = proved_theorem.traced_tactics[0]
        self.assertEqual(loaded_tactic.tactic, "rw [add_comm]")
        self.assertEqual(loaded_tactic.annotated_tactic, ("rw [add_comm]", []))
        self.assertEqual(loaded_tactic.state_before, "")
        self.assertEqual(loaded_tactic.state_after, "")

    def test_prove_sorry_theorems(self):
        results = [
            SearchResult(
                theorem=self.repo.sorry_theorems_unproved[0],
                status=Status.PROVED,
                proof=["rw [add_comm]", "refl"],
                actor_time=1.0,
                environment_time=2.0,
                total_time=3.0,
                num_total_nodes=10,
                num_searched_nodes=5
            )
        ]
        result = results[0] if results else None

        if isinstance(result, SearchResult) and result.status == Status.PROVED:
            logger.info("Proving theorem")
            traced_tactics = []
            for tactic in result.proof:
                traced_tactics.append(
                    AnnotatedTactic(
                        tactic=tactic,
                        annotated_tactic=(tactic, []),
                        state_before="",
                        state_after=""
                    )
                )
            self.repo.sorry_theorems_unproved[0].traced_tactics = traced_tactics
            self.repo.change_sorry_to_proven(self.repo.sorry_theorems_unproved[0])

        self.assertEqual(len(self.repo.sorry_theorems_unproved), 0)
        self.assertEqual(len(self.repo.sorry_theorems_proved), 1)
        proved_theorem = self.repo.sorry_theorems_proved[0]
        self.assertEqual(len(proved_theorem.traced_tactics), 2)
        self.assertEqual(proved_theorem.traced_tactics[0].tactic, "rw [add_comm]")
        self.assertEqual(proved_theorem.traced_tactics[1].tactic, "refl")

    def test_save_load_dynamic_database(self):
        json_file = "temp_file.json"

        self.db.to_json(json_file)
        loaded_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(self.db.repositories), len(loaded_db.repositories))
        self.assertEqual(self.db.repositories[0].url, loaded_db.repositories[0].url)
        self.assertEqual(self.db.repositories[0].commit, loaded_db.repositories[0].commit)
        self.assertEqual(len(self.db.repositories[0].sorry_theorems_unproved),
                            len(loaded_db.repositories[0].sorry_theorems_unproved))
    
    def test_add_repository_and_save(self):
        json_file = "temp_file.json"

        self.db.to_json(json_file)

        new_repo_data = {
            "url": "https://github.com/test/new-repo",
            "name": "new_test_repo",
            "commit": "1234567890abcdef",
            "lean_version": "4.0.0",
            "lean_dojo_version": "1.0.0",
            "date_processed": datetime.datetime.now().isoformat(),
            "metadata": {"key": "new_value"},
            "sorry_theorems_unproved": [
                {
                    "full_name": "new_test_theorem",
                    "file_path": "src/new_test.lean",
                    "start": [1, 1],
                    "end": [10, 1],
                    "url": "https://github.com/test/new-repo",
                    "commit": "1234567890abcdef",
                    "theorem_statement": "theorem new_test_theorem : 3 + 3 = 6 := sorry"
                }
            ]
        }

        new_repo = Repository.from_dict(new_repo_data)
        self.db.add_repository(new_repo)
        self.db.to_json(json_file)
        loaded_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(loaded_db.repositories), 2)
        self.assertEqual(loaded_db.repositories[1].url, "https://github.com/test/new-repo")
        self.assertEqual(len(loaded_db.repositories[1].sorry_theorems_unproved), 1)
        self.assertEqual(loaded_db.repositories[1].sorry_theorems_unproved[0].full_name, "new_test_theorem")

    def test_prove_sorry_theorems_and_save(self):
        json_file = "temp_file.json"
        
        self.db.to_json(json_file)

        results = [
            SearchResult(
                theorem=self.repo.sorry_theorems_unproved[0],
                status=Status.PROVED,
                proof=["rw [add_comm]", "refl"],
                actor_time=1.0,
                environment_time=2.0,
                total_time=3.0,
                num_total_nodes=10,
                num_searched_nodes=5
            )
        ]
        result = results[0] if results else None

        if isinstance(result, SearchResult) and result.status == Status.PROVED:
            logger.info("Proving theorem")
            traced_tactics = []
            for tactic in result.proof:
                traced_tactics.append(
                    AnnotatedTactic(
                        tactic=tactic,
                        annotated_tactic=(tactic, []),
                        state_before="",
                        state_after=""
                    )
                )
            self.repo.sorry_theorems_unproved[0].traced_tactics = traced_tactics
            self.repo.change_sorry_to_proven(self.repo.sorry_theorems_unproved[0])

        self.db.to_json(json_file)
        loaded_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(loaded_db.repositories[0].sorry_theorems_unproved), 0)
        self.assertEqual(len(loaded_db.repositories[0].sorry_theorems_proved), 1)
        proved_theorem = loaded_db.repositories[0].sorry_theorems_proved[0]
        self.assertEqual(proved_theorem.full_name, "test_theorem")
        self.assertEqual(len(proved_theorem.traced_tactics), 2)
        self.assertEqual(proved_theorem.traced_tactics[0].tactic, "rw [add_comm]")
        self.assertEqual(proved_theorem.traced_tactics[1].tactic, "refl")

if __name__ == '__main__':
    unittest.main()
