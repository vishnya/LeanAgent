# If curriculum learning is enabled, initialize repositories and sort them by difficulty
        if curriculum_learning:
            logger.info("Starting curriculum learning")
            repo_info_file = f"{RAID_DIR}/{DATA_DIR}/repo_info_compatible.json"  # TODO: make constnat?
            if is_main_process:
                search_github_repositories("Lean", num_repos)
                for i in range(len(lean_git_repos)):
                    repo = lean_git_repos[i]
                    logger.info(f"Processing {repo.url}")
                    result = add_repo_to_database(dynamic_database_json_path, repo, db)
                    if result is not None:
                        logger.info(f"Successfully added repo {repo.url}")                    
                logger.info(f"Successfully added {num_repos} repositories to the database")
                
                sorted_repos, categorized_theorems, percentiles = sort_repositories_by_difficulty(db)
                print("Sorted repositories. Saving now...")
                db.to_json(dynamic_database_json_path)
                save_sorted_repos(sorted_repos, "sorted_repos.json")
                print("Summary of theorem difficulties by URL:")
                for repo in sorted_repos:
                    print(f"\nURL: {repo.url}")
                    for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
                        theorems = categorized_theorems[repo][category]
                        print(f"  {category}: {len(theorems)} theorems")
                        if theorems:
                            sorted_theorems = sorted(theorems, key=lambda x: x[2] if x[2] is not None else -float('inf'), reverse=True)[:3]
                            for name, path, start, end, diff in sorted_theorems:
                                diff_str = f"{diff:.2f}" if diff is not None else "N/A"
                                print(f"    - {name} (File: {path}, Difficulty: {diff_str})")

                print("\nOverall Statistics:")
                total_theorems = sum(len(theorems) for categories in categorized_theorems.values() for theorems in categories.values())
                for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
                    count = sum(len(categories[category]) for categories in categorized_theorems.values())
                    percentage = (count / total_theorems) * 100
                    print(f"{category}: {count} theorems ({percentage:.2f}%)")

                print(f"\nPercentile thresholds: Easy <= {percentiles[0]:.2f}, Medium <= {percentiles[1]:.2f}, Hard > {percentiles[1]:.2f}")
            
                logger.info("Finding compatible repositories...")
                updated_repos = find_and_save_compatible_commits(repo_info_file, sorted_repos)
                lean_git_repos = [LeanGitRepo(repo['url'], repo['commit']) for repo in updated_repos]
                logger.info("Finished finding compatible repositories")

            # All processes wait for the file to be created and then read from it
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    with open(repo_info_file, 'r') as f:
                        repo_info = json.load(f)
                    break
                except (json.JSONDecodeError, FileNotFoundError):
                    if attempt == max_attempts - 1:
                        raise Exception("Failed to read repository information after multiple attempts")
                    time.sleep(1)
                
            # Load compatible repositories
            lean_git_repos = [LeanGitRepo(info['url'].replace('.git', ''), info['commit']) for info in repo_info]

            # Iterate over each repository and lambda value
            for i in range(num_repos):
                for lambda_value in lambdas:
                    logger.info(f"length of lean_git_repos: {len(lean_git_repos)}")
                    logger.info(f"i: {i}")
                    repo = lean_git_repos[i]
                    sha = repo.commit
                    dir_name = repo.url.split("/")[-1] + "_" + sha
                    result = True
                    if is_main_process:
                        logger.info("Main process")
                        logger.info(f"Using lambda = {lambda_value}")
                        logger.info(f"Processing {repo.url}")

                        if single_repo:
                            repos_for_merged_dataset = []

                        # TODO: don't always do merged_, if we change this then change the if condition in average test accordingly
                        # Create a directory for the merged dataset if it doesn't exist
                        dst_dir = Path(RAID_DIR) / DATA_DIR / f"merged_with_new_{dir_name}"
                        if (repo.url, repo.commit) not in repos_for_merged_dataset:
                            logger.info("Adding repo to repos_for_merged_dataset")
                            repos_for_merged_dataset.append((repo.url, repo.commit))
                        else:
                            logger.info("Repo already in repos_for_merged_dataset")

                        db.generate_merged_dataset(dst_dir, repos_for_merged_dataset)
                    
                    # TODO: reduce repition later with all path
                    dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + f"merged_with_new_{dir_name}"
                    new_data_path = dst_dir

                    logger.info("All GPUs")
                    model_checkpoint_path = None
                    best_model = None
                    data_module = None
                    if run_progressive_training:
                        try:
                            model_checkpoint_path = find_latest_checkpoint()
                            logger.info(f"Found latest checkpoint: {model_checkpoint_path}")
                        except FileNotFoundError as e:
                            logger.error(str(e))
                            model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                        
                        # Train the model on the new dataset that we generated from the dynamic database.
                        logger.info("Inside train_test_fisher")
                        logger.info(f"Starting training at epoch {current_epoch}")
                        seed_everything(3407)

                        # Progessive Training
                        
                        if not torch.cuda.is_available():
                            logger.warning("Indexing the corpus using CPU can be very slow.")
                            device = torch.device("cpu")
                        else:
                            device = torch.device("cuda")

                        # TODO: reduce repetition in code like this
                        config = {
                            "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
                            "lr": 1e-3,
                            "warmup_steps": 1000,
                            "max_seq_len": 512,
                            "num_retrieved": 100,
                        }

                        model = PremiseRetriever.load(
                            model_checkpoint_path, device, freeze=False, config=config
                        )
                        model.train()
                        logger.info(f"Loaded premise retriever at {model_checkpoint_path}")

                        # Load previous Fisher Information Matrix for current EWC
                        if use_fisher:
                            latest_fisher = find_latest_fisher()
                            fisher_info = load_fisher_information(latest_fisher)
                            model.set_fisher_info(fisher_info)
                            logger.info("Fisher Information Matrix loaded.")

                        # Initialize ModelCheckpoint and EarlyStopping
                        # TODO: use the yaml file instead of repeating here, same throughout
                        dir_name = new_data_path.split("/")[-1]
                        filename_suffix = f"_lambda_{lambda_value}"
                        checkpoint_callback = ModelCheckpoint(
                            dirpath=RAID_DIR + "/" + CHECKPOINT_DIR,
                            filename=dir_name + filename_suffix + "_{epoch}-{Recall@10_val:.2f}",
                            verbose=True,
                            save_top_k=-1,  # Save all checkpoints
                            every_n_epochs=1,  # Save every epoch (which is just once in this case)
                            monitor="Recall@10_val",
                            mode="max"
                        )
                        
                        early_stop_callback = EarlyStopping(
                            monitor="Recall@10_val",
                            patience=5,
                            mode="max",
                            verbose=True
                        )

                        lr_monitor = LearningRateMonitor(logging_interval='step')

                        # Set up environment variables for NCCL
                        VERY_LONG_TIMEOUT = 7 * 24 * 60 * 60 * 52  # 1 year
                        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
                        os.environ['NCCL_TIMEOUT'] = str(VERY_LONG_TIMEOUT * 1000)

                        # Create a custom log directory for Lightning
                        custom_log_dir = os.path.join(RAID_DIR, "lightning_logs", f"{dir_name}_{use_fisher}_lambda_{lambda_value}")
                        os.makedirs(custom_log_dir, exist_ok=True)

                        # Initialize DDP strategy
                        ddp_strategy = DDPStrategy(timeout=timedelta(seconds=VERY_LONG_TIMEOUT))
                        trainer = pl.Trainer(
                            accelerator="gpu",
                            gradient_clip_val=1.0,
                            precision="bf16-mixed",
                            strategy=ddp_strategy,
                            devices=4, # TODO: change for GPU
                            accumulate_grad_batches=4,
                            callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
                            max_epochs=current_epoch + epochs_per_repo,
                            log_every_n_steps=1,
                            num_sanity_val_steps=0,
                            default_root_dir=custom_log_dir,
                        )

                        # Barrier before data module
                        logger.info("right before barrier for data module")
                        trainer.strategy.barrier()
                        should_skip, skip_repo_url = should_skip_repo()
                        if should_skip:
                            logger.info(f"Skipping repository {skip_repo_url} due to preprocessing issues")
                            trainer.strategy.barrier()
                            if is_main_process:
                                logger.info("Removing skip file")
                                skip_file_path = os.path.join(RAID_DIR, DATA_DIR, "skip_repo.txt")
                                os.remove(skip_file_path)
                            continue

                        # Set lambda value for the model
                        model.set_lambda(lambda_value)
                        corpus_path = new_data_path + "/corpus.jsonl"
                        data_path = new_data_path + "/random"
                        logger.info(f"Data path: {data_path}")
                        data_module = RetrievalDataModule(
                            data_path=data_path,
                            corpus_path=corpus_path,
                            num_negatives=3,
                            num_in_file_negatives=1,
                            model_name="google/byt5-small",
                            batch_size=BATCH_SIZE,
                            eval_batch_size=64,
                            max_seq_len=1024,
                            num_workers=4
                        )
                        data_module.setup(stage='fit')

                        logger.info(f"Training dataset size after load: {len(data_module.ds_train)}")
                        logger.info(f"Validation dataset size after load: {len(data_module.ds_val)}")
                        logger.info(f"Testing dataset size after load: {len(data_module.ds_pred)}")

                        logger.info(f"Starting progressive training from epoch {current_epoch} to {current_epoch + epochs_per_repo}")

                        # Train the model
                        try:
                            logger.info("hit the barrier before training")
                            trainer.strategy.barrier()
                            trainer.fit(model, datamodule=data_module, ckpt_path=model_checkpoint_path)
                            logger.info("hit the barrier after training")
                            trainer.strategy.barrier()
                        except Exception as e:
                            print(f"An error occurred during training: {str(e)}")
                            print(traceback.format_exc())

                        logger.info(f"Finished progressive training at epoch {trainer.current_epoch}")

                        # Testing for Average Recall

                        try:
                            best_model_path = find_latest_checkpoint()
                            logger.info(f"Found latest checkpoint: {best_model_path}")
                            best_model = PremiseRetriever.load(best_model_path, device, freeze=False, config=config)
                        except FileNotFoundError as e:
                            logger.error(f"No checkpoint found: {str(e)}")
                            logger.warning("Using the current model state.")
                            best_model = model

                        best_model.eval()

                        logger.info("Testing...")
                        total_R1, total_R10, total_MRR = [], [], []
                        dataset_path = RAID_DIR + "/" + DATA_DIR
                        testing_paths = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)]
                        if is_main_process:
                            with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                                f.write("\n\n\n")
                                f.write(f"Results for {dir_name} with lambda = {lambda_value}")
                        for data_path in testing_paths:
                            # TODO: remove this for tests that do not use merged dataset
                            if "merged" not in data_path:
                                continue
                            
                            run_cli(best_model_path, data_path)
                            if is_main_process:
                                num_gpus = 4 # TODO: change for GPU
                                preds_map = {}
                                for gpu_id in range(num_gpus):
                                    with open(f"test_pickle_{gpu_id}.pkl", "rb") as f:
                                        preds = pickle.load(f)
                                        preds_map.update(preds)

                                logger.info("Loaded the predictions pickle files")
                                data_path = os.path.join(data_path, "random", "test.json")
                                data = json.load(open(data_path))
                                logger.info(f"Evaluating on {data_path}")
                                R1, R10, MRR = _eval(data, preds_map)
                                logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
                                total_R1.append(R1)
                                total_R10.append(R10)
                                total_MRR.append(MRR)
                                with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                                    f.write("\n\n\n")
                                    f.write(f"Intermediate results for {data_path}")
                                    f.write("\n\n\n")
                                    f.write(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")

                        if is_main_process:
                            avg_R1 = np.mean(total_R1)
                            avg_R10 = np.mean(total_R10)
                            avg_MRR = np.mean(total_MRR)

                            logger.info(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")

                            if not os.path.exists(EVAL_RESULTS_FILE_PATH):
                                open(EVAL_RESULTS_FILE_PATH, 'w').close()

                            with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                                f.write("\n\n\n")
                                f.write(f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}")
                    else:
                        model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                        if result is None:
                            logger.info(f"Skipping repository {repo.url} due to preprocessing issues")
                            continue

                    if is_main_process and run_progressive_training and use_fisher:
                        logger.info("Calculating Fisher Information Matrix for EWC")
                        # Fisher Information Matrix for Next EWC

                        # Switch to one GPU for calculating the Fisher Information Matrix
                        # TODO: barrier here
                        try:
                            # TODO: have separate intermediate checkpoints and save the epoch and data point, same for fisher
                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                            best_model.to(device)
                            train_dataloader = data_module.train_dataloader()
                            fisher_info = best_model.compute_fisher_information(train_dataloader, RAID_DIR + "/" + FISHER_DIR)
                            dir_path = RAID_DIR + "/" + FISHER_DIR
                            fisher_name = dir_path + "/" + dir_name + "_fisher_info.pkl"
                            with open(fisher_name, "wb") as f:
                                pickle.dump(fisher_info, f)
                            logger.info(f"Fisher info saved to {fisher_name}")
                        except Exception as e:
                            print(f"An error occurred during fisher: {str(e)}")
                            print(traceback.format_exc())

                    if is_main_process:
                        logger.info("Starting the prover")

                        if ray.is_initialized():
                            logger.info("Shutting down Ray before proving")
                            ray.shutdown()

                        # Set up the prover
                        use_vllm = False
                        corpus_path = dst_dir + "/corpus.jsonl"
                        tactic = None  # `None` since we are not using a fixed tactic generator
                        module = None  # `None` since we are not using a fixed tactic generator
                        num_workers = 4
                        num_gpus = 4 # TODO: change for GPU
                        timeout = 600
                        max_expansions = None
                        num_sampled_tactics = 64
                        debug = False
                        ckpt_path = f"{RAID_DIR}/model_lightning.ckpt"
                        prover = DistributedProver(
                            use_vllm,
                            ckpt_path,
                            corpus_path,
                            tactic,
                            module,
                            num_workers,
                            num_gpus=num_gpus,
                            timeout=timeout,
                            max_expansions=max_expansions,
                            num_sampled_tactics=num_sampled_tactics,
                            raid_dir=RAID_DIR,
                            checkpoint_dir=CHECKPOINT_DIR,
                            debug=debug,
                            run_progressive_training=run_progressive_training
                        )

                        # Prove sorry theorems
                        prove_sorry_theorems(db, prover, dynamic_database_json_path, repos_for_merged_dataset)
                        db.to_json(dynamic_database_json_path)

                        logger.info("Finished searching for proofs of sorry theorems")

                        if ray.is_initialized():
                            logger.info("Shutting down Ray after proving")
                            ray.shutdown()

                        # TODO: need to return proofs
                        # proofs = []
                        # Uncomment if you would like to contribute back to the repos!
                        # else:
                        #     base_branch = get_default_branch(repo_no_dir)
                        #     subprocess.run(["git", "-C", repo, "fetch", "origin", base_branch], check=True)
                        #     subprocess.run(["git", "-C", repo, "checkout", base_branch], check=True)
                        #     subprocess.run(["git", "-C", repo, "pull", "origin", base_branch], check=True)
                        #     create_or_switch_branch(repo, TMP_BRANCH, base_branch)
                        #     replace_sorry_with_proof(proofs)
                        #     committed = commit_changes(repo, COMMIT_MESSAGE)
                        #     if committed:
                        #         push_changes(repo, TMP_BRANCH)
                        #         url = str(create_pull_request(repo_no_dir, PR_TITLE, PR_BODY, TMP_BRANCH))
                        #         # TODO: add the PR URL to the database
                        #     shutil.rmtree(repo)
                    
                    logger.info("Finished processing the repository")
                    current_epoch += epochs_per_repo
                    logger.info(f"current epoch: {current_epoch}")
