import builtins
import json
from collections import defaultdict
from typing import Any
from utils.scraping_utils import scrape_url
from ddgs import DDGS
from openai import AsyncOpenAI, OpenAI

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

from utils.scraping_utils import web_search

DEFAULT_VLLM_BASE_URL = "http://rack-gamir-g11.cs.tau.ac.il:8000/v1"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on a topic. Often this is a very large string so you are encouraged to iteratively process it.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}


class QwenClient(BaseLM):
    """
    LM Client for running models served locally via vLLM.
    Uses the OpenAI-compatible API exposed by vLLM.
    Supports built-in web search via tool calls.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        base_url: str = DEFAULT_VLLM_BASE_URL,
        enable_search: bool = True,
        enable_thinking: bool = False,
        search_max_results: int = 5,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = base_url
        self.enable_search = enable_search
        self.enable_thinking = enable_thinking
        self.search_max_results = search_max_results

        self.client = OpenAI(base_url=base_url, api_key="dummy", timeout=self.timeout)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key="dummy", timeout=self.timeout)

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def _to_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def _handle_tool_calls(self, messages: list[dict], message) -> list[dict]:
        """Execute tool calls and append results to messages."""
        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = web_search(args["query"], self.search_max_results)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        return messages

    def _run_loop(self, messages: list[dict[str, Any]]) -> tuple[str, Any]:
        """Synchronous agentic loop handling tool calls until a final response."""
        tools = [SEARCH_TOOL] if self.enable_search else []
        extra_body = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

        while True:
            kwargs = dict(
                model=self.model_name,
                messages=messages,
                extra_body=extra_body,
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            self._track_usage(response)

            if finish_reason == "stop" or not message.tool_calls:
                return message.content, response

            messages = self._handle_tool_calls(messages, message)

    async def _run_loop_async(self, messages: list[dict[str, Any]]) -> tuple[str, Any]:
        """Async agentic loop handling tool calls until a final response."""
        tools = [SEARCH_TOOL] if self.enable_search else []
        extra_body = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

        while True:
            kwargs = dict(
                model=self.model_name,
                messages=messages,
                extra_body=extra_body,
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self.async_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            self._track_usage(response)

            if finish_reason == "stop" or not message.tool_calls:
                return message.content, response

            messages = self._handle_tool_calls(messages, message)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if model:
            self.model_name = model
        messages = self._to_messages(prompt)
        result, _ = self._run_loop(messages)
        return result

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if model:
            self.model_name = model
        messages = self._to_messages(prompt)
        result, _ = await self._run_loop_async(messages)
        return result

    def _track_usage(self, response):
        model = self.model_name
        self.model_call_counts[model] += 1
        usage = response.usage
        if usage:
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0
            self.model_input_tokens[model] += input_tokens
            self.model_output_tokens[model] += output_tokens
            self.last_prompt_tokens = input_tokens
            self.last_completion_tokens = output_tokens
        else:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
