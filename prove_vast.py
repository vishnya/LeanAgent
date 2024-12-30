
def prove_sorry_theorems(db: DynamicDatabase, prover: DistributedProver, dynamic_database_json_path, repos_to_include: Optional[List[Tuple[str, str]]] = None):
    repos_to_process = db.repositories if repos_to_include is None else [
        repo for repo in db.repositories if (repo.url, repo.commit) in repos_to_include
    ]

    # To avoid proving the same theorem multiple times, potentially from different versions of the
    # same repo, we sort the repositories
    repos_to_process.sort(key=lambda r: r.metadata['date_processed'], reverse=True)

    processed_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = set()
    all_encountered_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = set()
    last_save_time = datetime.datetime.now()
    save_interval = timedelta(minutes=30)

    for repo in repos_to_process:
        sorry_theorems = repo.sorry_theorems_unproved
        repo_url = repo.url
        repo_commit = repo.commit

        logger.info(f"Found {len(sorry_theorems)} sorry theorems to prove")
    
        for theorem in tqdm(sorry_theorems, desc=f"Processing theorems from {repo.name}", unit="theorem"):
            # Ignore sorry theorems from the repo's dependencies
            if theorem.url != repo_url or theorem.commit != repo_commit:
                continue

            theorem_id = theorem_identifier(theorem)
            all_encountered_theorems.add(theorem_id)
            if theorem_id in processed_theorems:
                logger.info(f"Skipping already processed theorem: {theorem.full_name}")
                continue

            processed_theorems.add(theorem_id)
            
            logger.info(f"Searching for proof for {theorem.full_name}")
            logger.info(f"Position: {theorem.start}")

            # Convert our Theorem to LeanDojo Theorem
            lean_dojo_theorem = LeanDojoTheorem(
                repo=LeanGitRepo(repo_url, repo_commit),
                file_path=theorem.file_path,
                full_name=theorem.full_name
            )

            results = prover.search_unordered(LeanGitRepo(repo_url, repo_commit), [lean_dojo_theorem], [Pos(*theorem.start)])
            result = results[0] if results else None

            if isinstance(result, SearchResult) and result.status == Status.PROVED:
                logger.info(f"Proof found for {theorem.full_name}")

                # Convert the proof to AnnotatedTactic objects
                # We have to simplify some of the fields since LeanDojo does not
                # prvoide all the necessary information
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
                
                theorem.traced_tactics = traced_tactics
                repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)
                db.update_repository(repo)
                db.to_json(dynamic_database_json_path)

                logger.info(f"Updated theorem {theorem.full_name} in the database")
            else:
                logger.info(f"No proof found for {theorem.full_name}")
            
            current_time = datetime.datetime.now()
            if current_time - last_save_time >= save_interval:
                logger.info("Saving encountered theorems...")
                with open(ENCOUNTERED_THEOREMS_FILE, 'wb') as f:
                    pickle.dump(all_encountered_theorems, f)
                last_save_time = current_time

    logger.info("Final save to JSON file...")
    with open(ENCOUNTERED_THEOREMS_FILE, 'wb') as f:
        pickle.dump(all_encountered_theorems, f)

    logger.info("Finished attempting to prove sorry theorems")
