# TODO: edge cases where items are missing

import datetime
from pathlib import Path
from lean_dojo.data_extraction.lean import Pos
from dynamic_database import DynamicDatabase, Repository, Theorem, AnnotatedTactic, Annotation, PremiseFile, Premise

def create_sample_database():
    db = DynamicDatabase()
    
    repo = Repository(
        url="https://github.com/example/repo",
        name="Example Repo",
        commit="abc123",
        lean_version="3.50.3",
        date_processed=datetime.datetime.now(),
        metadata={"key": "value"},
        total_theorems=3
    )

    theorem1 = Theorem(
        name="Example Theorem",
        statement="∀ x y : ℕ, x + y = y + x",
        file_path=Path("src/example.lean"),
        full_name="example.theorem1",
        start=Pos(1, 1),
        end=Pos(5, 10),
        traced_tactics=[
            AnnotatedTactic(
                tactic="induction x",
                annotated_tactic=("induction x", [
                    Annotation(
                        full_name="induction",
                        def_path="src/tactic/induction.lean",
                        def_pos=Pos(100, 1),
                        def_end_pos=Pos(100, 10)
                    )
                ]),
                state_before="⊢ ∀ x y : ℕ, x + y = y + x",
                state_after="2 goals\ncase zero\n⊢ ∀ y : ℕ, 0 + y = y + 0\ncase succ\n⊢ ∀ y : ℕ, succ n + y = y + succ n"
            )
        ],
        difficulty_rating=0.7
    )

    theorem2 = Theorem(
        name="Sorry Theorem",
        statement="∀ x : ℝ, x^2 ≥ 0",
        file_path=Path("src/sorry_example.lean"),
        full_name="sorry_example.theorem2",
        start=Pos(10, 1),
        end=Pos(12, 10),
        traced_tactics=[],
        difficulty_rating=0.5
    )

    repo.proven_theorems.append(theorem1)
    repo.sorry_theorems_unproved.append(theorem2)

    premise_file = PremiseFile(
        path=Path("src/premise.lean"),
        imports=["import data.nat.basic"],
        premises=[
            Premise(
                full_name="add_zero",
                code="theorem add_zero (n : ℕ) : n + 0 = n := rfl",
                start=Pos(1, 1),
                end=Pos(1, 50),
                kind="theorem"
            )
        ]
    )

    repo.premise_files.append(premise_file)
    repo.files_traced.append(Path("src/example.lean"))

    db.add_repository(repo)
    return db

def test_add_get_update_repository():
    db = DynamicDatabase()
    
    repo1 = Repository(
        url="https://github.com/example/repo1",
        name="Example Repo 1",
        commit="abc123",
        lean_version="3.50.3",
        date_processed=datetime.datetime.now(),
        metadata={"key": "value1"},
        total_theorems=1
    )
    
    db.add_repository(repo1)
    assert len(db.repositories) == 1
    
    retrieved_repo = db.get_repository("https://github.com/example/repo1", "abc123")
    assert retrieved_repo is not None
    assert retrieved_repo.name == "Example Repo 1"
    
    repo2 = Repository(
        url="https://github.com/example/repo2",
        name="Example Repo 2",
        commit="def456",
        lean_version="3.50.3",
        date_processed=datetime.datetime.now(),
        metadata={"key": "value2"},
        total_theorems=2
    )
    
    db.add_repository(repo2)
    assert len(db.repositories) == 2
    
    updated_repo1 = Repository(
        url="https://github.com/example/repo1",
        name="Updated Example Repo 1",
        commit="abc123",
        lean_version="3.50.3",
        date_processed=datetime.datetime.now(),
        metadata={"key": "updated_value"},
        total_theorems=3
    )
    
    db.update_repository(updated_repo1)
    assert len(db.repositories) == 2
    
    retrieved_updated_repo = db.get_repository("https://github.com/example/repo1", "abc123")
    assert retrieved_updated_repo is not None
    assert retrieved_updated_repo.name == "Updated Example Repo 1"
    assert retrieved_updated_repo.metadata["key"] == "updated_value"
    assert retrieved_updated_repo.total_theorems == 3

def test_serialization_deserialization_and_modification():
    original_db = create_sample_database()
    
    # Serialize to JSON
    json_file = "test_database.json"
    original_db.to_json(json_file)
    
    # Deserialize from JSON
    deserialized_db = DynamicDatabase.from_json(json_file)
    
    # Modify the deserialized database
    repo = deserialized_db.get_repository("https://github.com/example/repo", "abc123")
    assert repo is not None
    
    # Find the sorry theorem
    sorry_theorem = next((thm for thm in repo.sorry_theorems_unproved if thm.name == "Sorry Theorem"), None)
    assert sorry_theorem is not None
    
    # Write a proof for the sorry theorem
    sorry_theorem.traced_tactics = [
        AnnotatedTactic(
            tactic="by_cases h : x = 0",
            annotated_tactic=("by_cases h : x = 0", []),
            state_before="⊢ ∀ x : ℝ, x^2 ≥ 0",
            state_after="2 goals\ncase pos\nh : x = 0\n⊢ x^2 ≥ 0\ncase neg\nh : x ≠ 0\n⊢ x^2 ≥ 0"
        ),
        AnnotatedTactic(
            tactic="{ rw h, simp }",
            annotated_tactic=("{ rw h, simp }", []),
            state_before="h : x = 0\n⊢ x^2 ≥ 0",
            state_after="no goals"
        ),
        AnnotatedTactic(
            tactic="{ apply le_of_lt, apply mul_self_pos, exact h }",
            annotated_tactic=("{ apply le_of_lt, apply mul_self_pos, exact h }", []),
            state_before="h : x ≠ 0\n⊢ x^2 ≥ 0",
            state_after="no goals"
        )
    ]
    
    # Move the theorem from sorry_theorems_unproved to proven_theorems
    repo.sorry_theorems_unproved.remove(sorry_theorem)
    repo.proven_theorems.append(sorry_theorem)
    
    # Update the JSON file with the modified database
    deserialized_db.update_json(json_file)
    
    # Read the updated JSON file
    updated_db = DynamicDatabase.from_json(json_file)
    updated_repo = updated_db.get_repository("https://github.com/example/repo", "abc123")
    assert updated_repo is not None
    
    # Check if the sorry theorem has been moved and has a proof
    assert len(updated_repo.sorry_theorems_unproved) == 0
    assert len(updated_repo.proven_theorems) == 2
    
    updated_theorem = next((thm for thm in updated_repo.proven_theorems if thm.name == "Sorry Theorem"), None)
    assert updated_theorem is not None
    assert len(updated_theorem.traced_tactics) == 3
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_add_get_update_repository()
    test_serialization_deserialization_and_modification()