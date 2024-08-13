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
    repo.sorry_theorems_proved.append(theorem2)

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

def test_serialization_deserialization():
    original_db = create_sample_database()
    
    # Serialize to JSON
    json_file = "test_database.json"
    original_db.to_json(json_file)
    
    # Deserialize from JSON
    deserialized_db = DynamicDatabase.from_json(json_file)
    
    # Compare original and deserialized databases
    assert len(original_db.repositories) == len(deserialized_db.repositories)
    
    original_repo = original_db.repositories[0]
    deserialized_repo = deserialized_db.repositories[0]
    
    assert original_repo.url == deserialized_repo.url
    assert original_repo.name == deserialized_repo.name
    assert original_repo.commit == deserialized_repo.commit
    assert original_repo.lean_version == deserialized_repo.lean_version
    assert original_repo.date_processed.isoformat() == deserialized_repo.date_processed.isoformat()
    assert original_repo.metadata == deserialized_repo.metadata
    assert original_repo.total_theorems == deserialized_repo.total_theorems
    
    assert len(original_repo.proven_theorems) == len(deserialized_repo.proven_theorems)
    assert len(original_repo.sorry_theorems_proved) == len(deserialized_repo.sorry_theorems_proved)
    
    original_theorem = original_repo.proven_theorems[0]
    deserialized_theorem = deserialized_repo.proven_theorems[0]
    
    assert original_theorem.name == deserialized_theorem.name
    assert original_theorem.statement == deserialized_theorem.statement
    assert str(original_theorem.file_path) == str(deserialized_theorem.file_path)
    assert original_theorem.full_name == deserialized_theorem.full_name
    assert repr(original_theorem.start) == repr(deserialized_theorem.start)
    assert repr(original_theorem.end) == repr(deserialized_theorem.end)
    assert original_theorem.difficulty_rating == deserialized_theorem.difficulty_rating
    
    assert len(original_theorem.traced_tactics) == len(deserialized_theorem.traced_tactics)
    
    original_tactic = original_theorem.traced_tactics[0]
    deserialized_tactic = deserialized_theorem.traced_tactics[0]
    
    assert original_tactic.tactic == deserialized_tactic.tactic
    assert original_tactic.annotated_tactic[0] == deserialized_tactic.annotated_tactic[0]
    assert len(original_tactic.annotated_tactic[1]) == len(deserialized_tactic.annotated_tactic[1])
    assert original_tactic.state_before == deserialized_tactic.state_before
    assert original_tactic.state_after == deserialized_tactic.state_after
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_serialization_deserialization()