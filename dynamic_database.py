# TODO: Think about use case and add more accordingly
# TODO: Test 
# TODO: Write unit tests
# TODO: Check which of these fields can be empty. Reference the source code.
# TODO: add validation. For example, len(proven) = len(sorry_proved) + len(sorry_unproved)

from __future__ import annotations
import datetime
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from lean_dojo.data_extraction.lean import Pos

@dataclass
class Annotation:
    full_name: str
    def_path: str
    def_pos: Pos
    def_end_pos: Pos

@dataclass
class AnnotatedTactic:
    tactic: str
    annotated_tactic: Tuple[str, List[Annotation]]
    state_before: str
    state_after: str

@dataclass
class Theorem:
    name: str
    statement: str
    file_path: Path
    full_name: str
    start: Pos
    end: Pos
    traced_tactics: Optional[List[AnnotatedTactic]] = None
    difficulty_rating: Optional[float] = None

@dataclass
class Premise:
    full_name: str
    code: str
    start: Pos
    end: Pos
    kind: str

@dataclass
class PremiseFile:
    path: Path
    imports: List[str]
    premises: List[Premise]

@dataclass
class Repository:
    url: str
    name: str
    commit: str
    lean_version: str
    date_processed: datetime.datetime
    metadata: Dict[str, str]
    total_theorems: int
    proven_theorems: List[Theorem] = field(default_factory=list)
    sorry_theorems_proved: List[Theorem] = field(default_factory=list)
    sorry_theorems_unproved: List[Theorem] = field(default_factory=list)
    premise_files: List[PremiseFile] = field(default_factory=list)
    files_traced: List[Path] = field(default_factory=list)
    pr_url: Optional[str] = None

    @property
    def num_proven_theorems(self) -> int:
        return len(self.proven_theorems)

    @property
    def num_sorry_theorems_proved(self) -> int:
        return len(self.sorry_theorems_proved)

    @property
    def num_sorry_theorems_unproved(self) -> int:
        return len(self.sorry_theorems_unproved)

    @property
    def num_sorry_theorems(self) -> int:
        return self.num_sorry_theorems_proved + self.num_sorry_theorems_unproved

    @property
    def total_premises(self) -> int:
        return sum(len(pf.premises) for pf in self.premise_files)

    @property
    def num_files_traced(self) -> int:
        return len(self.files_traced)

@dataclass
class DynamicDatabase:
    repositories: List[Repository] = field(default_factory=list)

    def add_repository(self, repo: Repository) -> None:
        self.repositories.append(repo)

    def get_repository(self, url: str, commit: str) -> Optional[Repository]:
        for repo in self.repositories:
            if repo.url == url and repo.commit == commit:
                return repo
        return None

    def update_repository(self, repo: Repository) -> None:
        for i, existing_repo in enumerate(self.repositories):
            if existing_repo.url == repo.url and existing_repo.commit == repo.commit:
                self.repositories[i] = repo
                return
        self.add_repository(repo)

    def to_dict(self) -> Dict:
        return {
            "repositories": [
                {
                    "url": repo.url,
                    "name": repo.name,
                    "commit": repo.commit,
                    "lean_version": repo.lean_version,
                    "date_processed": repo.date_processed.isoformat(),
                    "metadata": repo.metadata,
                    "total_theorems": repo.total_theorems,
                    "proven_theorems": [
                        {
                            "name": thm.name,
                            "statement": thm.statement,
                            "file_path": str(thm.file_path),
                            "full_name": thm.full_name,
                            "start": repr(thm.start),
                            "end": repr(thm.end),
                            "traced_tactics": [
                                {
                                    "tactic": t.tactic,
                                    "annotated_tactic": [
                                        t.annotated_tactic[0],
                                        [
                                            {
                                                "full_name": a.full_name,
                                                "def_path": a.def_path,
                                                "def_pos": repr(a.def_pos),
                                                "def_end_pos": repr(a.def_end_pos)
                                            } for a in t.annotated_tactic[1]
                                        ]
                                    ],
                                    "state_before": t.state_before,
                                    "state_after": t.state_after
                                } for t in (thm.traced_tactics or [])
                            ],
                            "difficulty_rating": thm.difficulty_rating
                        } for thm in repo.proven_theorems
                    ],
                    "sorry_theorems_proved": [
                        {
                            "name": thm.name,
                            "statement": thm.statement,
                            "file_path": str(thm.file_path),
                            "full_name": thm.full_name,
                            "start": repr(thm.start),
                            "end": repr(thm.end),
                            "traced_tactics": [
                                {
                                    "tactic": t.tactic,
                                    "annotated_tactic": [
                                        t.annotated_tactic[0],
                                        [
                                            {
                                                "full_name": a.full_name,
                                                "def_path": a.def_path,
                                                "def_pos": repr(a.def_pos),
                                                "def_end_pos": repr(a.def_end_pos)
                                            } for a in t.annotated_tactic[1]
                                        ]
                                    ],
                                    "state_before": t.state_before,
                                    "state_after": t.state_after
                                } for t in (thm.traced_tactics or [])
                            ],
                            "difficulty_rating": thm.difficulty_rating
                        } for thm in repo.sorry_theorems_proved
                    ],
                    "sorry_theorems_unproved": [
                        {
                            "name": thm.name,
                            "statement": thm.statement,
                            "file_path": str(thm.file_path),
                            "full_name": thm.full_name,
                            "start": repr(thm.start),
                            "end": repr(thm.end),
                            "difficulty_rating": thm.difficulty_rating
                        } for thm in repo.sorry_theorems_unproved
                    ],
                    "premise_files": [
                        {
                            "path": str(pf.path),
                            "imports": pf.imports,
                            "premises": [
                                {
                                    "full_name": premise.full_name,
                                    "code": premise.code,
                                    "start": repr(premise.start),
                                    "end": repr(premise.end),
                                    "kind": premise.kind,
                                } for premise in pf.premises
                            ]
                        } for pf in repo.premise_files
                    ],
                    "files_traced": [str(file) for file in repo.files_traced],
                    "pr_url": repo.pr_url
                } for repo in self.repositories
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> DynamicDatabase:
        db = cls()
        for repo_data in data.get("repositories", []):
            repo = Repository(
                url=repo_data["url"],
                name=repo_data["name"],
                commit=repo_data["commit"],
                lean_version=repo_data["lean_version"],
                date_processed=datetime.datetime.fromisoformat(repo_data["date_processed"]),
                metadata=repo_data["metadata"],
                total_theorems=repo_data["total_theorems"],
                proven_theorems=[
                    Theorem(
                        name=thm["name"],
                        statement=thm["statement"],
                        file_path=Path(thm["file_path"]),
                        full_name=thm["full_name"],
                        start=Pos.from_str(thm["start"]),
                        end=Pos.from_str(thm["end"]),
                        traced_tactics=[
                            AnnotatedTactic(
                                tactic=t["tactic"],
                                annotated_tactic=(
                                    t["annotated_tactic"][0],
                                    [
                                        Annotation(
                                            full_name=a["full_name"],
                                            def_path=a["def_path"],
                                            def_pos=Pos.from_str(a["def_pos"]),
                                            def_end_pos=Pos.from_str(a["def_end_pos"])
                                        ) for a in t["annotated_tactic"][1]
                                    ]
                                ),
                                state_before=t["state_before"],
                                state_after=t["state_after"]
                            ) for t in thm.get("traced_tactics", [])
                        ],
                        difficulty_rating=thm.get("difficulty_rating")
                    ) for thm in repo_data.get("proven_theorems", [])
                ],
                sorry_theorems_proved=[
                    Theorem(
                        name=thm["name"],
                        statement=thm["statement"],
                        file_path=Path(thm["file_path"]),
                        full_name=thm["full_name"],
                        start=Pos.from_str(thm["start"]),
                        end=Pos.from_str(thm["end"]),
                        traced_tactics=[
                            AnnotatedTactic(
                                tactic=t["tactic"],
                                annotated_tactic=(
                                    t["annotated_tactic"][0],
                                    [
                                        Annotation(
                                            full_name=a["full_name"],
                                            def_path=a["def_path"],
                                            def_pos=Pos.from_str(a["def_pos"]),
                                            def_end_pos=Pos.from_str(a["def_end_pos"])
                                        ) for a in t["annotated_tactic"][1]
                                    ]
                                ),
                                state_before=t["state_before"],
                                state_after=t["state_after"]
                            ) for t in thm.get("traced_tactics", [])
                        ],
                        difficulty_rating=thm.get("difficulty_rating")
                    ) for thm in repo_data.get("sorry_theorems_proved", [])
                ],
                sorry_theorems_unproved=[
                    Theorem(
                        name=thm["name"],
                        statement=thm["statement"],
                        file_path=Path(thm["file_path"]),
                        full_name=thm["full_name"],
                        start=Pos.from_str(thm["start"]),
                        end=Pos.from_str(thm["end"]),
                        difficulty_rating=thm.get("difficulty_rating")
                    ) for thm in repo_data.get("sorry_theorems_unproved", [])
                ],
                premise_files=[
                    PremiseFile(
                        path=Path(pf["path"]),
                        imports=pf["imports"],
                        premises=[
                            Premise(
                                full_name=premise["full_name"],
                                code=premise["code"],
                                start=Pos.from_str(premise["start"]),
                                end=Pos.from_str(premise["end"]),
                                kind=premise["kind"],
                            ) for premise in pf["premises"]
                        ]
                    ) for pf in repo_data.get("premise_files", [])
                ],
                files_traced=[Path(file) for file in repo_data.get("files_traced", [])],
                pr_url=repo_data.get("pr_url")
            )
            db.add_repository(repo)
        return db

    def to_json(self, file_path: str) -> None:
        """Serialize the database to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, file_path: str) -> DynamicDatabase:
        """Deserialize the database from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update_json(self, file_path: str) -> None:
        """Update an existing JSON file with the current database state."""
        try:
            existing_db = self.from_json(file_path)
        except FileNotFoundError:
            existing_db = DynamicDatabase()

        for repo in self.repositories:
            existing_db.update_repository(repo)

        existing_db.to_json(file_path)