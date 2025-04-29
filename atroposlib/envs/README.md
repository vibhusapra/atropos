# Base Environment (`BaseEnv`)

The `BaseEnv` class (located in `trajectoryhandler/envs/base.py`) provides a foundation for creating custom reinforcement learning environments that interact with Atropos. When creating your own environment, you will typically subclass `BaseEnv` and implement several key methods.

## Core Methods to Implement

These methods **must** be implemented in your subclass:

*   **`async def setup(self)`**: This method is called once at the beginning of the environment's lifecycle (`env_manager`). Use it for any initial setup required for your specific environment, such as loading datasets, initializing models, or connecting to external resources.

*   **`async def get_next_item(self) -> Item`**: This method is responsible for generating or retrieving the next piece of data (prompt, state, etc.) that will be used to start a new trajectory collection. If no more items are available or should be generated, it can return `None` to signal the worker to pause.

*   **`async def collect_trajectory(self, item: Item) -> Tuple[Any | None, List[Item]]`**: This method defines the logic for a *single* logical trajectory collection step based on the input `item`. \
    *   **How it relates to multiple generations**: The `BaseEnv` uses `collect_trajectories` to run this method multiple times in parallel (controlled by `group_size`) to gather a batch of trajectories. \
    *   **Your implementation**: You can implement this method to generate *one* response/trajectory per call.\
    *   **Return value**: It returns a tuple containing:\
        1.  The collected data for this step (one trajectory). This data can be processed further in `postprocess_histories`, if you require additional filtering right before sending to the API.\
        2.  A list of new `Item` objects to be added to the backlog for future processing (e.g., follow-up prompts).\

*   **`async def evaluate(self, *args, **kwargs)`**: This method is called periodically (controlled by `steps_per_eval` in the config) to perform evaluation runs. You define the evaluation logic here. The base class provides an example using `self.eval_workers` for parallel evaluation tasks, but you can implement any evaluation procedure suitable for your environment.

## Optional Methods to Override

These methods have default implementations or are optional based on your needs:

*   **`async def collect_trajectories(self, item: Item) -> Tuple[Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any | None]], List[Item]]`**: The default implementation of this method runs `collect_trajectory` multiple times in parallel (controlled by `group_size`). You can override this *instead* of `collect_trajectory` if you have a more efficient way to generate the entire group of responses/trajectories at once based on the input `item`. It should return the collected group data and a list of backlog items.

*   **`async def postprocess_histories(self, trajectories: Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]`**: This method is called after `collect_trajectories` and before the data is sent to the training server. It receives the collected data from the parallel runs (or your custom `collect_trajectories` implementation). Use this to perform final processing, scoring, or formatting you may require before sending to the server. You usually won't need this.

*   **`async def wandb_log(self, wandb_metrics: Optional[Dict] = None)`**: Called periodically to log metrics to Weights & Biases. If you override this to add custom metrics, **ensure you call `super().wandb_log(wandb_metrics)`** at the end of your implementation. This ensures that the base class's performance metrics and rollout tables are also logged.
    ```python
    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        # Add your custom metrics
        wandb_metrics['my_custom_metric'] = calculate_my_metric()
        # ... add more metrics

        # Call the parent method to log base metrics
        await super().wandb_log(wandb_metrics)
    ```

*   **`save_checkpoint(self, step, data=None)`**: The base class calls this method automatically at checkpoint intervals determined by the server. It saves the provided `data` dictionary (which you might populate with environment-specific state) to a JSON file. You can override this to customize *what* data is saved or *how* it's saved (e.g., using a different format or location), but the triggering mechanism remains automatic.

*   **`@classmethod config_init(cls) -> Tuple[BaseEnvConfig, Union[ServerBaseline, List[OpenaiConfig]]]`**: This class method is used by the default `get_cli_serve_config_cls` implementation to get the initial environment configuration (`BaseEnvConfig` subclass) and server configurations (`ServerBaseline` or `List[OpenaiConfig]`) when setting up the `serve` command. The default implementation returns `cls.env_config_cls(), ServerBaseline()`. You might override this if your environment requires different default configurations or specific server setups (like multiple `OpenaiConfig` instances) when run via the CLI `serve` command.

*   **`async def cleanup(self)`**: Called after each call to `handle_env`. You can implement this for any cleanup needed after processing a single item, though it's often not required.

## Provided Functionality

`BaseEnv` provides several helpful features:

*   **Parallel Trajectory Collection (`collect_trajectories`)**: The base implementation runs your `collect_trajectory` method multiple times in parallel (based on `group_size`) and gathers the results. You can override `collect_trajectories` directly for custom group generation logic (see Optional Methods).
*   **Server Interaction**: Handles registration with the rollout server, fetching configuration (like `batch_size`), sending scored data (`handle_send_to_api` with retries), and status updates.
*   **WandB Integration**: Sets up WandB logging (if enabled) based on server information and provides the `wandb_log` hook for custom metrics (remember to call `super().wandb_log()`). It uses helper methods `add_rollouts_for_wandb` (to temporarily store rollout data) and `create_rollout_table` (to format the data into a `wandb.Table`). You can override either of these helpers for custom logging behavior (e.g., changing what data is stored or how the final table is structured).
*   **Checkpointing**:
    *   The environment automatically triggers checkpoint saves based on the `checkpoint_interval` received from the server, calling the `save_checkpoint` method (see Optional Methods).
    *   `load_checkpoint(self)`: Loads data from the checkpoint file corresponding to the environment's `curr_step`. It attempts to restore attributes of the environment object based on the keys in the loaded JSON data. This is called automatically if `curr_step > 0` during registration.
*   **Worker Management**: Manages asynchronous worker tasks for collecting trajectories (`add_train_workers`, `handle_env`).
*   **Performance Monitoring**: Tracks and logs various performance statistics (task durations, worker counts, etc.).
*   **CLI Integration**: Provides a `cli()` class method using `pydantic-cli` to easily create command-line interfaces for your environment (e.g., `python your_env_module.py serve --port 8001 ...`). See `get_cli_serve_config_cls` and `get_cli_process_config_cls`.

By implementing the required methods and optionally overriding others, you can create diverse environments that leverage the distributed training infrastructure provided by the `Atropos` framework.
