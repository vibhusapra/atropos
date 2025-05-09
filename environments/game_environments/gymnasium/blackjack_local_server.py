import argparse
import asyncio
import logging
import os
import random

from dotenv import load_dotenv
from environments.game_environments.gymnasium.blackjack_env import BlackjackEnv, BlackjackEnvConfig
from atroposlib.envs.base import OpenaiConfig, EvalHandlingEnum

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def parse_arguments(): # Removed
#     parser = argparse.ArgumentParser(description="Blackjack environment local server")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="blackjack_local",
#         help="Configuration file name (without .yaml extension, relative to "
#         "envs/gymnasium/configs), or full path to a YAML file.",
#     )
#     return parser.parse_args()


async def main():
    logger.info("Starting Blackjack environment local debug runner")

    # args = parse_arguments() # Removed

    # Removed logic for config_name_or_path and BlackjackEnv.config_init

    # Create hardcoded configurations for local debugging
    env_config = BlackjackEnvConfig(
        # BaseEnvConfig fields, tailored for debug
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=1,  # Debug single generation path
        use_wandb=False,
        wandb_name="blackjack_local_debug", # Explicitly set for debug
        max_num_workers=1,
        rollout_server_url="http://localhost:8000", # Standard default
        total_steps=1,
        batch_size=1, # Consistent with 1 step, 1 worker, group_size 1
        steps_per_eval=0, # No eval steps needed
        max_token_length=1024 * 4, # Reduced for faster local debugging if necessary
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE, # No evaluation in this script
        eval_limit_ratio=0.0,

        # BlackjackEnvConfig specific fields (from blackjack_env.py's definition or defaults)
        env_name="Blackjack-v1",
        temperature=0.2, # Lower temperature for more deterministic debug output
        top_p=0.9,       # Standard default
        max_turns=5,     # Standard default
        thinking_active=True,
        eval_episodes=0, # No evaluation episodes
        max_think_chars_history=3000,
        max_trajectory_tokens=24576,
        debug_mode=True, # Enable debug logging from the environment
        mc_samples=1,    # With group_size=1, this means 1 MC rollout for V(s)
    )

    server_configs = [
        OpenaiConfig(
            model_name="gpt-4.1-mini", # Ensure this is locally available if not mocked
            base_url="https://api.openai.com/v1", # Explicitly set OpenAI base URL
            api_key=os.getenv("OPENAI_API_KEY"), # Use env var or default
            num_requests_for_eval=0, # No eval requests
        )
    ]
    logger.info("Using hardcoded debug configuration.")
    logger.debug(f"Env Config: {env_config}")
    logger.debug(f"Server Configs: {server_configs}")

    # Create and set up the environment using the loaded configs
    try:
        env = BlackjackEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False,  # Explicitly false for local testing
        )
    except Exception as e:
        logger.exception(f"Failed to initialize BlackjackEnv: {e}")
        return

    # Run a single trajectory directly
    logger.info("Running a single trajectory directly")
    try:
        await env.setup()  # Setup the server connection etc.
        seed = random.randint(0, 1000000)
        logger.info(f"Using seed: {seed}")

        # Make sure the episode exists before collecting
        # This also initializes the message history correctly
        _ = env._get_or_create_episode(seed)

        result_trajectory = await env.collect_trajectory(seed)
        logger.info(
            f"Trajectory collection complete with {len(result_trajectory)} steps."
        )

        episode_summary = None
        if env.completed_episode_metrics_buffer:
            # Assume the last entry in the buffer corresponds to the trajectory just run
            episode_summary = env.completed_episode_metrics_buffer[-1]
            # Optionally, clear the buffer if this script is only for single runs
            # env.completed_episode_metrics_buffer.clear()

        if episode_summary and episode_summary.get("seed") == seed:
            # Print a final summary
            logger.info("\n========== Episode Summary ==========")
            logger.info(f"Seed: {episode_summary['seed']}")
            logger.info(f"Total steps taken: {episode_summary['num_steps']}")
            logger.info(
                f"Final Environment reward: {episode_summary['total_reward']:.2f}"
            )

            game_outcome_val = episode_summary.get("game_outcome", 0)
            outcome_str = "Draw"
            if game_outcome_val == 1:
                outcome_str = "Win"
            elif game_outcome_val == -1:
                outcome_str = "Loss"
            logger.info(
                f"Game Outcome: {outcome_str} (Reward: {episode_summary['total_reward']:.0f})"
            )

            # Calculate and log action accuracy based on EpisodeState fields
            if episode_summary["num_total_actions"] > 0:
                accuracy = episode_summary["num_correct_actions"] / max(
                    1, episode_summary["num_total_actions"]
                )
                logger.info(
                    f"Action accuracy (valid tool calls): "
                    f"{episode_summary['num_correct_actions']}/{episode_summary['num_total_actions']} "
                    f"({accuracy:.2%})"
                )
            else:
                logger.info(
                    "Action accuracy (valid tool calls): No tool calls attempted or recorded."
                )
            logger.info("=======================================")
        else:
            logger.error(
                f"Could not get episode summary for seed {seed} from metrics buffer or seed mismatch."
            )

    except Exception as e:
        logger.exception(
            f"An error occurred during trajectory collection or summary: {e}"
        )


if __name__ == "__main__":
    asyncio.run(main())
