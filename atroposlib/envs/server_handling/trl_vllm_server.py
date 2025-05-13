"""
This is a server that interfaces with trl's vLLM server.

Developed with much help from @winglian when they worked on integrating Atropos into Axolotl.
"""

import asyncio
import time
import uuid
from typing import Optional

import aiohttp
import numpy as np
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import Completion
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer

from atroposlib.envs.server_handling.openai_server import AsyncSemWithAdaptiveWeight
from atroposlib.envs.server_handling.server_baseline import APIServerConfig


class TrlVllmServer:
    """
    A server that interfaces with trl's vLLM server.
    """

    def __init__(self, config: APIServerConfig):
        self.config = config
        self.sem = AsyncSemWithAdaptiveWeight(config.num_max_requests_at_once)
        self.eval_sem = AsyncSemWithAdaptiveWeight(config.num_requests_for_eval)
        self.server_healthy = True
        self.attempts_list = []
        self.request_timings = []
        # in case eval is much different, we should keep different buffers
        self.eval_attempts_list = []
        self.eval_request_timings = []
        self.check_task = None
        self.initialized = False
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    async def update_weight(self, weight: float) -> None:
        # need to update sems
        self.sem.update_weight(weight)
        self.eval_sem.update_weight(weight)

    async def check_server_status_task(self):
        # TODO: Implement server health check for trl's vLLM server
        self.server_healthy = True

    async def wandb_metrics(
        self, metrics_dict: Optional[dict], server_name: Optional[str]
    ):
        if server_name is None:
            server_name = "server"
        if len(self.request_timings) > 0:
            metrics_dict[f"server/{server_name}_request_time_avg"] = np.mean(
                self.request_timings
            )
            metrics_dict[f"server/{server_name}_request_time_std"] = np.std(
                self.request_timings
            )
            metrics_dict[f"server/{server_name}_request_time_99p"] = np.percentile(
                self.request_timings, 99
            )
        if len(self.eval_request_timings) > 0:
            metrics_dict[f"server/{server_name}_eval_request_time_avg"] = np.mean(
                self.eval_request_timings
            )
            metrics_dict[f"server/{server_name}_eval_request_time_std"] = np.std(
                self.eval_request_timings
            )
            metrics_dict[f"server/{server_name}_eval_request_time_99p"] = np.percentile(
                self.eval_request_timings, 99
            )
        if len(self.attempts_list) > 0:
            metrics_dict[f"server/{server_name}_average_num_attempts"] = np.mean(
                self.attempts_list
            )
        if len(self.eval_attempts_list) > 0:
            metrics_dict[f"server/{server_name}_eval_retry_rate"] = np.mean(
                self.eval_attempts_list
            )
        return metrics_dict

    async def _chat_handle(self, **kwargs) -> ChatCompletion:
        url = f"{self.config.base_url}/generate/"
        prompt = kwargs.get("messages", [])
        prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "prompts": [prompt],
                    "n": kwargs.get("n", 1),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                    "temperature": kwargs.get("temperature", 1.0),
                    "top_p": kwargs.get("top_p", 1.0),
                    "top_k": kwargs.get("top_k", -1),
                    "min_p": kwargs.get("min_p", 0.0),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                },
            ) as response:
                completions = await response.json()
        completions = ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                Choice(
                    finish_reason=(
                        "stop"
                        if self.tokenizer.eos_token_id in completion
                        else "length"
                    ),
                    index=i,
                    message=ChatCompletionMessage(
                        content=self.tokenizer.decode(completion),
                        role="assistant",
                    ),
                )
                for i, completion in enumerate(completions["completion_ids"])
            ],
        )
        return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _chat_comp(self, stat_dict, **kwargs) -> ChatCompletion:
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.sem:
            print(kwargs)
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._chat_handle(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _chat_eval(self, stat_dict, **kwargs) -> ChatCompletion:
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.eval_sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._chat_handle(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def chat_completion(self, **kwargs) -> ChatCompletion:
        if not self.initialized:
            if (
                self.config.base_url is not None
            ):  # skip health check if using OpenAI API
                self.check_task = asyncio.create_task(self.check_server_status_task())
            else:
                self.server_healthy = True
            self.initialized = True
        kwargs["model"] = self.config.model_name
        split = kwargs.pop("split", "train")
        stat_dict = {}
        stat_dict["attempts"] = 0
        if split == "train":
            ret_data = await self._chat_comp(stat_dict, **kwargs)
            self.request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.attempts_list.append(stat_dict["attempts"])
        else:
            # Give separate eval workers, if desired, gotta go fast for those evals
            ret_data = await self._chat_eval(stat_dict, **kwargs)
            self.eval_request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.eval_attempts_list.append(stat_dict["attempts"])
        return ret_data

    async def _comp_handle(self, **kwargs) -> ChatCompletion:
        url = f"{self.config.base_url}/generate/"
        prompt = kwargs.get("prompt", "")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "prompts": [prompt],
                    "n": kwargs.get("n", 1),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                    "temperature": kwargs.get("temperature", 1.0),
                    "top_p": kwargs.get("top_p", 1.0),
                    "top_k": kwargs.get("top_k", -1),
                    "min_p": kwargs.get("min_p", 0.0),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                },
            ) as response:
                completions = await response.json()
        completions = ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                Choice(
                    finish_reason=(
                        "stop"
                        if self.tokenizer.eos_token_id in completion
                        else "length"
                    ),
                    index=i,
                    message=ChatCompletionMessage(
                        content=self.tokenizer.decode(completion),
                        role="assistant",
                    ),
                )
                for i, completion in enumerate(completions["completion_ids"])
            ],
        )
        return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _comp(self, stat_dict, **kwargs) -> Completion:
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._comp_handle(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _comp_eval(self, stat_dict, **kwargs) -> Completion:
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.eval_sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._comp_handle(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    async def completion(self, **kwargs) -> Completion:
        if not self.initialized:
            if (
                self.config.base_url is not None
            ):  # skip health check if using OpenAI API
                self.check_task = asyncio.create_task(self.check_server_status_task())
            else:
                self.server_healthy = True
            self.initialized = True
        kwargs["model"] = self.config.model_name
        split = kwargs.pop("split", "train")
        stat_dict = {}
        stat_dict["attempts"] = 0
        if split == "train":
            ret_data = await self._comp(stat_dict, **kwargs)
            self.request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.attempts_list.append(stat_dict["attempts"])
        else:
            # Give separate eval workers, if desired, gotta go fast for those evals
            ret_data = await self._comp_eval(stat_dict, **kwargs)
            self.eval_request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.eval_attempts_list.append(stat_dict["attempts"])
        return ret_data
