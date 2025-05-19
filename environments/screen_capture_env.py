import base64
from typing import List, Optional, Tuple

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class ScreenCaptureEnv(BaseEnv):
    """Environment for training on screen capture videos.

    The environment presents a screen recording alongside a textual task
    description. A multimodal model must produce a concise explanation of
    the actions in the video. Rewards are given based on how well the
    explanation matches the expected steps.
    """

    name = "screen_capture"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = []
        self.iter = 0

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        config = BaseEnvConfig(
            wandb_name="screen_capture",
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=4,
            use_wandb=True,
            max_num_workers=2,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=8,
            steps_per_eval=100,
            max_token_length=2048,
        )

        server_configs = [
            APIServerConfig(
                model_name="gemini-2.5-pro",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]

        return config, server_configs

    async def setup(self):
        """Initialize a simple in-memory dataset."""
        self.dataset = [
            {
                "video_path": "sample_videos/open_browser.mp4",
                "task": "Explain how the user opens a browser and navigates to example.com.",
                "expected_steps": [
                    "open the browser",
                    "go to example.com",
                ],
            }
        ]
        self.train = self.dataset
        self.iter = 0

    def _load_video_base64(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            return ""

    async def get_next_item(self) -> Item:
        entry = self.train[self.iter % len(self.train)]
        self.iter += 1
        video_b64 = self._load_video_base64(entry["video_path"])
        prompt = tuple([frozenset({"role": "user", "content": entry["task"]}.items())])
        return (prompt, entry["expected_steps"], video_b64)

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        prompt_tuple, expected_steps, video_b64 = item
        text_prompt = dict(prompt_tuple[0])["content"]
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant that explains the actions in a screen recording step by step.",
        }
        if video_b64:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                    },
                ],
            }
        else:
            user_msg = {"role": "user", "content": text_prompt}

        messages = [system_msg, user_msg]
        completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=512,
            timeout=120,
        )

        to_score: List[Tuple[GameHistory, List[str], str]] = []
        for choice in completions.choices:
            history = (
                {"role": "user", "content": text_prompt},
                {"role": "assistant", "content": choice.message.content},
            )
            to_score.append((history, expected_steps, video_b64))

        scored = await self.score(to_score)
        return scored, []

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        for history, expected_steps, _ in rollout_group_data:
            out = tokenize_for_trainer(self.tokenizer, history)
            assistant_answer = history[-1]["content"].lower()
            matches = 0
            for step in expected_steps:
                if step.lower() in assistant_answer:
                    matches += 1
            reward = matches / len(expected_steps)
            scores["tokens"].append(out["tokens"])
            scores["masks"].append(out["masks"])
            scores["scores"].append(reward)
        return scores

    async def evaluate(self, *args, **kwargs):
        metrics = {}
        eval_scores = []
        for entry in self.train:
            prompt = (
                tuple([frozenset({"role": "user", "content": entry["task"]}.items())]),
                entry["expected_steps"],
                self._load_video_base64(entry["video_path"]),
            )
            scored, _ = await self.collect_trajectories(prompt)
            if scored and scored["scores"]:
                eval_scores.extend(scored["scores"])
        if eval_scores:
            metrics["eval/mean_score"] = sum(eval_scores) / len(eval_scores)
        await self.wandb_log(metrics)


if __name__ == "__main__":
    ScreenCaptureEnv.cli()
