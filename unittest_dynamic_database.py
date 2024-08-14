import unittest
import datetime
from pathlib import Path
from dynamic_database import DynamicDatabase, Repository, Theorem, AnnotatedTactic, Annotation, PremiseFile, Premise
from lean_dojo.data_extraction.lean import Pos, LeanGitRepo
import generate_benchmark_lean4
import lean_dojo
import json
import shutil

RAID_DIR = "/raid/adarsh"
DATA_DIR = "datasets_new"

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
                train_set = set(item['full_name'] for item in train_data)

            with open(dst_dir / strategy / "val.json", 'r') as f:
                val_data = json.load(f)
                val_set = set(item['full_name'] for item in val_data)

            with open(dst_dir / strategy / "test.json", 'r') as f:
                test_data = json.load(f)
                test_set = set(item['full_name'] for item in test_data)

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

if __name__ == '__main__':
    unittest.main()
