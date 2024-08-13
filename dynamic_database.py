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
    lean_dojo_version: str
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
    def num_premise_files(self) -> int:
        return len(self.premise_files)

    @property
    def num_premises(self) -> int:
        return sum(len(pf.premises) for pf in self.premise_files)

    @property
    def num_files_traced(self) -> int:
        return len(self.files_traced)
    
    @property
    def get_all_theorems(self) -> List[Theorem]:
        return self.proven_theorems + self.sorry_theorems_proved + self.sorry_theorems_unproved

    def add_theorem(self, theorem: Theorem) -> None:
        if not theorem.traced_tactics:  # Theorem is proven with a term-style proof
            if theorem not in self.proven_theorems:
                self.proven_theorems.append(theorem)
                self.total_theorems += 1
        elif any(step.tactic == 'sorry' for step in theorem.traced_tactics):
            if theorem not in self.sorry_theorems_unproved:
                self.sorry_theorems_unproved.append(theorem)
                self.total_theorems += 1
        else:
            if theorem not in self.sorry_theorems_proved:
                self.sorry_theorems_proved.append(theorem)
                self.total_theorems += 1

    def get_theorem(self, file_path: str, full_name: str) -> Optional[Theorem]:
        for thm in self.get_all_theorems:
            if thm.file_path == file_path and thm.full_name == full_name:
                return thm
        return None

    def update_theorem(self, updated_theorem: Theorem) -> None:
        for thm_list in [self.proven_theorems, self.sorry_theorems_proved, self.sorry_theorems_unproved]:
            for i, thm in enumerate(thm_list):
                if thm.file_path == updated_theorem.file_path and thm.full_name == updated_theorem.full_name:
                    thm_list[i] = updated_theorem
                    return
        raise ValueError(f"Theorem '{updated_theorem.full_name}' not found.")

    def delete_theorem(self, file_path: str, full_name: str) -> None:
        for thm_list in [self.proven_theorems, self.sorry_theorems_proved, self.sorry_theorems_unproved]:
            for i, thm in enumerate(thm_list):
                if thm.file_path == file_path and thm.full_name == full_name:
                    del thm_list[i]
                    self.total_theorems -= 1
                    return
        raise ValueError(f"Theorem '{full_name}' not found.")

    def add_premise_file(self, premise_file: PremiseFile) -> None:
        if premise_file not in self.premise_files:
            self.premise_files.append(premise_file)

    def get_premise_file(self, path: str) -> Optional[PremiseFile]:
        for pf in self.premise_files:
            if str(pf.path) == path:
                return pf
        return None

    def update_premise_file(self, updated_premise_file: PremiseFile) -> None:
        for i, pf in enumerate(self.premise_files):
            if pf.path == updated_premise_file.path:
                self.premise_files[i] = updated_premise_file
                return
        raise ValueError(f"Premise file '{updated_premise_file.path}' not found.")

    def delete_premise_file(self, path: str) -> None:
        for i, pf in enumerate(self.premise_files):
            if str(pf.path) == path:
                del self.premise_files[i]
                return
        raise ValueError(f"Premise file '{path}' not found.")

    def add_traced_file(self, file_path: Path) -> None:
        if file_path not in self.files_traced:
            self.files_traced.append(file_path)

    def get_traced_file(self, file_path: str) -> Optional[Path]:
        path = Path(file_path)
        if path in self.files_traced:
            return path
        return None

    def update_traced_file(self, old_file_path: str, new_file_path: str) -> None:
        old_path = Path(old_file_path)
        new_path = Path(new_file_path)
        if old_path in self.files_traced:
            self.files_traced.remove(old_path)
            self.files_traced.append(new_path)
        else:
            raise ValueError(f"Traced file '{old_file_path}' not found.")

    def delete_traced_file(self, file_path: str) -> None:
        path = Path(file_path)
        if path in self.files_traced:
            self.files_traced.remove(path)
        else:
            raise ValueError(f"Traced file '{file_path}' not found.")

    def change_sorry_to_proven(self, theorem: Theorem) -> None:
        if theorem in self.sorry_theorems_unproved:
            self.sorry_theorems_unproved.remove(theorem)
            self.sorry_theorems_proved.append(theorem)
        else:
            raise ValueError("The theorem is not in the list of unproved sorry theorems.")

@dataclass
class DynamicDatabase:
    repositories: List[Repository] = field(default_factory=list)

    def add_repository(self, repo: Repository) -> None:
        if repo not in self.repositories:
            self.repositories.append(repo)

    def get_repository(self, url: str, commit: str) -> Optional[Repository]:
        for repo in self.repositories:
            if repo.url == url and repo.commit == commit:
                return repo
        return None

    def update_repository(self, updated_repo: Repository) -> None:
        for i, repo in enumerate(self.repositories):
            if repo.url == updated_repo.url and repo.commit == updated_repo.commit:
                self.repositories[i] = updated_repo
                return
        raise ValueError(f"Repository '{updated_repo.url}' with commit '{updated_repo.commit}' not found.")

    def delete_repository(self, url: str, commit: str) -> None:
        for i, repo in enumerate(self.repositories):
            if repo.url == url and repo.commit == commit:
                del self.repositories[i]
                return
        raise ValueError(f"Repository '{url}' with commit '{commit}' not found.")

    def to_dict(self) -> Dict:
        return {
            "repositories": [
                {
                    "url": repo.url,
                    "name": repo.name,
                    "commit": repo.commit,
                    "lean_version": repo.lean_version,
                    "lean_dojo_version": repo.lean_dojo_version,
                    "date_processed": repo.date_processed.isoformat(),
                    "metadata": repo.metadata,
                    "total_theorems": repo.total_theorems,
                    "num_proven_theorems": repo.num_proven_theorems,
                    "num_sorry_theorems": repo.num_sorry_theorems,
                    "num_sorry_theorems_proved": repo.num_sorry_theorems_proved,
                    "num_sorry_theorems_unproved": repo.num_sorry_theorems_unproved,
                    "num_premise_files": repo.num_premise_files,
                    "num_premises": repo.num_premises,
                    "num_files_traced": repo.num_files_traced,
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
                lean_dojo_version=repo_data["lean_dojo_version"],
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