# TODO: test changing Annotation, adding Theorem, changing Theorem, adding Repository, changing Repository, adding Premise, changing Premise, adding PremiseFile, changing PremiseFile, saving for all
# TODO: test all methods in Repository
# TODO: missing data
# TODO: repo with no theorems, no premises, no files_traced
# TODO: empty lists
# TODO: None for optional values
# TODO: empty string for required like name
# TODO: very large JSON may cause memory issues
# TODO: write unit tests

import datetime
from pathlib import Path
from lean_dojo.data_extraction.lean import Pos
from dynamic_database import DynamicDatabase, Repository, Theorem, AnnotatedTactic, Annotation, PremiseFile, Premise

def create_sample_database_with_unicode():
    db = DynamicDatabase()
    
    repo = Repository(
        url="https://github.com/example/repo",
        name="Example Repo with Unicode ユニコード",
        commit="abc123",
        lean_version="3.50.3",
        date_processed=datetime.datetime.now(),
        metadata={"key": "value with Unicode ✨"},
        total_theorems=3
    )

    theorem1 = Theorem(
        name="Commutativity of Addition",
        statement="∀ x y : ℕ, x + y = y + x",
        file_path=Path("src/example.lean"),
        full_name="example.commutative_addition",
        start=Pos(1, 1),
        end=Pos(5, 10),
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
        name="Quadratic Formula",
        statement="∀ a b c x : ℝ, a ≠ 0 → (a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a))",
        file_path=Path("src/sorry_example.lean"),
        full_name="example.quadratic_formula",
        start=Pos(10, 1),
        end=Pos(12, 10),
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

    db.add_repository(repo)
    return db

def test_unicode_serialization_deserialization():
    original_db = create_sample_database_with_unicode()
    
    # Serialize to JSON
    json_file = "test_unicode_database.json"
    original_db.to_json(json_file)
    
    # Deserialize from JSON
    deserialized_db = DynamicDatabase.from_json(json_file)
    
    # Compare original and deserialized databases
    assert len(original_db.repositories) == len(deserialized_db.repositories)
    
    original_repo = original_db.repositories[0]
    deserialized_repo = deserialized_db.repositories[0]
    
    assert original_repo.name == deserialized_repo.name
    assert original_repo.metadata["key"] == deserialized_repo.metadata["key"]
    
    original_theorem1 = original_repo.proven_theorems[0]
    deserialized_theorem1 = deserialized_repo.proven_theorems[0]
    
    assert original_theorem1.statement == deserialized_theorem1.statement
    assert original_theorem1.traced_tactics[0].state_before == deserialized_theorem1.traced_tactics[0].state_before
    assert original_theorem1.traced_tactics[0].state_after == deserialized_theorem1.traced_tactics[0].state_after
    
    original_theorem2 = original_repo.sorry_theorems_unproved[0]
    deserialized_theorem2 = deserialized_repo.sorry_theorems_unproved[0]
    
    assert original_theorem2.statement == deserialized_theorem2.statement
    
    original_premise = original_repo.premise_files[0].premises[0]
    deserialized_premise = deserialized_repo.premise_files[0].premises[0]
    
    assert original_premise.code == deserialized_premise.code

    print("Unicode serialization and deserialization test passed successfully!")

def test_unicode_modification():
    db = create_sample_database_with_unicode()
    
    # Serialize to JSON
    json_file = "test_unicode_database.json"
    db.to_json(json_file)
    
    # Deserialize from JSON
    deserialized_db = DynamicDatabase.from_json(json_file)
    
    # Modify the deserialized database
    repo = deserialized_db.get_repository("https://github.com/example/repo", "abc123")
    assert repo is not None
    
    # Find the sorry theorem
    sorry_theorem = repo.sorry_theorems_unproved[0]
    
    # Write a proof for the sorry theorem with Unicode
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
    
    # Move the theorem from sorry_theorems_unproved to sorry_theorems_proved
    repo.change_sorry_to_proven(sorry_theorem)
    
    # Update the JSON file with the modified database
    deserialized_db.update_json(json_file)
    
    # Read the updated JSON file
    updated_db = DynamicDatabase.from_json(json_file)
    updated_repo = updated_db.get_repository("https://github.com/example/repo", "abc123")
    assert updated_repo is not None
    
    # Check if the sorry theorem has been moved and has a proof
    assert len(updated_repo.sorry_theorems_unproved) == 0
    assert len(updated_repo.sorry_theorems_proved) == 1
    
    updated_theorem = updated_repo.sorry_theorems_proved[0]
    assert updated_theorem.name == "Quadratic Formula"
    assert len(updated_theorem.traced_tactics) == 3
    assert "√(b² - 4*a*c)" in updated_theorem.traced_tactics[0].state_before
    assert "↔" in updated_theorem.traced_tactics[1].state_before

    print("Unicode modification test passed successfully!")

if __name__ == "__main__":
    test_unicode_serialization_deserialization()
    test_unicode_modification()