import copy
import io
import json
import os
from datetime import datetime
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import (
    RESERVED_TOOL_NAMES,
    NonIsolatedEnv,
    extract_tool_value,
    validate_custom_tools,
)


# =============================================================================
# Safe Builtins
# =============================================================================

# Safe builtins - blocks dangerous operations like eval/exec/input
_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


# Keys that live in self.globals and should never be copied into self.locals
_GLOBALS_ONLY_KEYS: frozenset[str] = frozenset(
    {"__builtins__", "__name__", "FINAL_VAR", "SHOW_VARS",
     "llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"}
)

_TYPE_INSTRUCTIONS: dict[type, str] = {
    int: "Respond with ONLY an integer. No explanation, units, or other text.",
    float: "Respond with ONLY a decimal number. No explanation, units, or other text.",
    bool: "Respond with ONLY 'true' or 'false'. No explanation or other text.",
    list: "Respond with ONLY a valid JSON array. No explanation, markdown, or other text.",
    dict: "Respond with ONLY a valid JSON object. No explanation, markdown, or other text.",
    datetime: "Respond with ONLY a date in ISO 8601 format (YYYY-MM-DD). No explanation or other text.",
}


def _serialize_schema(schema: dict) -> str:
    """Serialize a schema dict to JSON, converting Python type objects to their names."""
    def _convert(v: Any) -> Any:
        if isinstance(v, type):
            return v.__name__
        if isinstance(v, dict):
            return {k2: _convert(v2) for k2, v2 in v.items()}
        if isinstance(v, list):
            return [_convert(item) for item in v]
        return v

    return json.dumps({k: _convert(v) for k, v in schema.items()})


def _build_typed_prompt(prompt: str, return_type: type | None, schema: dict | None) -> str:
    """Augment prompt with instructions for returning a specific native type."""
    if return_type is None or return_type is str:
        return prompt
    suffix = _TYPE_INSTRUCTIONS.get(return_type, "")
    if schema is not None and return_type in (list, dict):
        suffix += " The response must conform to this schema: " + _serialize_schema(schema)
    if suffix:
        prompt = prompt + "\n\nIMPORTANT: " + suffix
    return prompt


def _parse_typed_response(response: str, return_type: type | None) -> Any:
    """Parse LM response string into the requested native Python type."""
    if return_type is None or return_type is str:
        return response
    if response is None:
        raise ValueError("Response is None (model returned empty/blocked response)")
    s = response.strip()
    try:
        if return_type is int:
            return int(s)
        if return_type is float:
            return float(s)
        if return_type is bool:
            low = s.lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
            raise ValueError(f"Cannot parse as bool: {s!r}")
        if return_type is datetime:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        if return_type in (list, dict):
            # Strip markdown code blocks if present
            if "```" in s:
                lines = s.split("\n")
                json_lines: list[str] = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                extracted = "\n".join(json_lines).strip()
                if extracted:
                    s = extracted
            return json.loads(s)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse response as {return_type.__name__}: {e}") from e
    return response


class LocalREPL(NonIsolatedEnv):
    """
    Local REPL environment with persistent Python namespace.
    Executes code in a sandboxed namespace with access to context data.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        subcall_fn: Callable[[str, str | None], RLMChatCompletion] | None = None,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        **kwargs,
    ):
        super().__init__(persistent=persistent, depth=depth, **kwargs)

        self.lm_handler_address = lm_handler_address
        self.subcall_fn = subcall_fn  # Callback for recursive RLM calls (depth > 1 support)
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()
        self._context_count: int = 0
        self._history_count: int = 0
        self.compaction = compaction

        # Custom tools: functions available in the REPL
        self.custom_tools = custom_tools or {}
        # Sub-tools: inherited from custom_tools if not specified
        self.custom_sub_tools = (
            custom_sub_tools if custom_sub_tools is not None else self.custom_tools
        )

        # Validate custom tools don't override reserved names
        validate_custom_tools(self.custom_tools)

        # Setup globals, locals, and modules in environment.
        self.setup()

        if compaction:
            self._compaction_history: list[Any] = []
            self.locals["history"] = self._compaction_history

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Setup the environment."""
        # Create sandboxed globals
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}

        # Track LLM calls made during code execution
        self._pending_llm_calls: list[RLMChatCompletion] = []
        # When FINAL_VAR is called inside a REPL block, we store the value here for the main loop
        self._last_final_answer: str | None = None

        # Add helper functions
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["datetime"] = datetime
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query"] = self._rlm_query
        self.globals["rlm_query_batched"] = self._rlm_query_batched

        # Add custom tools to globals
        # Tools can be either plain values or (value, description) tuples
        for name, entry in self.custom_tools.items():
            value = extract_tool_value(entry)
            if callable(value):
                self.globals[name] = value
            else:
                # For non-callable values (constants, data), add to locals
                self.locals[name] = value

    def _final_var(self, variable_name: str | Any) -> str:
        """Return the value of a variable as a final answer for the main model, or stringify a direct value."""
        if not isinstance(variable_name, str):
            answer = str(variable_name)
            self._last_final_answer = answer
            return answer
        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            answer = str(self.locals[variable_name])
            self._last_final_answer = answer
            return answer

        # Provide helpful error message with available variables (do not set _last_final_answer)
        available = [k for k in self.locals.keys() if not k.startswith("_")]
        if available:
            return (
                f"Error: Variable '{variable_name}' not found. "
                f"Available variables: {available}. "
                f"You must create and assign a variable BEFORE calling FINAL_VAR on it."
            )
        return (
            f"Error: Variable '{variable_name}' not found. "
            f"No variables have been created yet. "
            f"You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it."
        )

    def _show_vars(self) -> str:
        """Show all available variables in the REPL environment."""
        available = {k: type(v).__name__ for k, v in self.locals.items() if not k.startswith("_")}
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _fallback_extract(self, raw_response: str, return_type: type, model: str | None) -> Any:
        """Make a secondary LLM call to extract a typed value from a verbose/malformed response.

        Called when _parse_typed_response raises ValueError. Asks the LLM to extract only
        the typed value from the raw text and re-parses. Raises ValueError if still fails.
        """
        instruction = _TYPE_INSTRUCTIONS.get(return_type, "")
        extraction_prompt = (
            f"The following text should contain a {return_type.__name__} value but is not "
            f"in the expected format. Extract only the {return_type.__name__} value and "
            f"return it with no extra text.\n{instruction}\n\nText: {raw_response}"
        )
        request = LMRequest(prompt=extraction_prompt, model=model, depth=self.depth)
        response = send_lm_request(self.lm_handler_address, request)
        if not response.success:
            raise ValueError(f"Fallback extraction failed: {response.error}")
        self._pending_llm_calls.append(response.chat_completion)
        # Raises ValueError if still unparseable — propagates to caller
        return _parse_typed_response(response.chat_completion.response, return_type)

    def _llm_query(
        self,
        prompt: str,
        model: str | None = None,
        *,
        return_type: type,
        schema: dict | None = None,
    ) -> Any:
        """Query the LM with a single plain completion (no REPL, no recursion).

        This always makes a direct LM call via the handler, regardless of depth.

        Args:
            prompt: The prompt to send to the LM.
            model: Optional model name to use (if handler has multiple clients).
            return_type: Optional native Python type to cast the response to (int, float,
                bool, str, list, dict). When specified, the prompt is augmented with
                formatting instructions and the response is parsed into that type.
            schema: Optional JSON schema dict used when return_type is list or dict,
                added to the prompt as a formatting constraint.
        """
        if not self.lm_handler_address:
            return "Error: No LM handler configured"

        try:
            typed_prompt = _build_typed_prompt(prompt, return_type, schema)
            request = LMRequest(prompt=typed_prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}"

            self._pending_llm_calls.append(response.chat_completion)
            raw = response.chat_completion.response
            try:
                return _parse_typed_response(raw, return_type)
            except ValueError:
                return self._fallback_extract(raw, return_type, model)
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(
        self,
        queries: list[tuple[str, type, dict | None]],
        model: str | None = None,
    ) -> list[Any]:
        """Query the LM with multiple prompts concurrently (no REPL, no recursion).

        This always makes direct LM calls via the handler, regardless of depth.

        Args:
            queries: List of (prompt, return_type, schema) tuples. Each tuple specifies
                the prompt text, the Python type to parse the response into (str, int,
                float, bool, list, dict), and an optional schema dict for list/dict types.
            model: Optional model name to use (if handler has multiple clients).

        Returns:
            List of responses in the same order as input queries, each parsed to its return_type.
        """
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(queries)
        try:
            typed_prompts = [_build_typed_prompt(p, rt, s) for p, rt, s in queries]
            responses = send_lm_request_batched(
                self.lm_handler_address, typed_prompts, model=model, depth=self.depth
            )

            results = []
            to_retry: list[tuple[int, str, type]] = []  # (index, raw_response, return_type)
            for i, (response, (_, rt, _)) in enumerate(zip(responses, queries)):
                if not response.success:
                    results.append(f"Error: {response.error}")
                else:
                    self._pending_llm_calls.append(response.chat_completion)
                    raw = response.chat_completion.response
                    try:
                        results.append(_parse_typed_response(raw, rt))
                    except ValueError:
                        results.append(None)  # placeholder, filled by fallback below
                        to_retry.append((i, raw, rt))

            # Batch fallback: re-ask LLM to extract typed values for failed parses
            if to_retry:
                # Separate None-raw items (blocked/empty response) from malformed ones
                real_retry = [(i, raw, rt) for i, raw, rt in to_retry if raw is not None]
                for i, raw, rt in to_retry:
                    if raw is None:
                        results[i] = f"Error: model returned empty/blocked response"

                if real_retry:
                    fallback_prompts = []
                    for (_, raw, rt) in real_retry:
                        instruction = _TYPE_INSTRUCTIONS.get(rt, "")
                        fallback_prompts.append(
                            f"The following text should contain a {rt.__name__} value but is not "
                            f"in the expected format. Extract only the {rt.__name__} value and "
                            f"return it with no extra text.\n{instruction}\n\nText: {raw}"
                        )
                    fallback_responses = send_lm_request_batched(
                        self.lm_handler_address, fallback_prompts, model=model, depth=self.depth
                    )
                    for (idx, _, rt), fb in zip(real_retry, fallback_responses):
                        if not fb.success:
                            raise ValueError(f"Fallback extraction failed: {fb.error}")
                        self._pending_llm_calls.append(fb.chat_completion)
                        results[idx] = _parse_typed_response(fb.chat_completion.response, rt)

            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(queries)

    def _rlm_query(self, prompt: str, model: str | None = None, *, return_type: type, schema: dict | None = None) -> Any:
        """Spawn a recursive RLM sub-call for deeper thinking on a subtask.

        When a subcall callback is available (max_depth > 1), this spawns a child
        RLM with its own REPL that can reason over the prompt iteratively.
        Falls back to a plain llm_query if no recursive capability is configured.

        Args:
            prompt: The prompt to send to the child RLM.
            model: Optional model name override for the child.
            return_type: Python type to parse the response into.
            schema: Optional schema dict for list/dict return types.
        """
        if self.subcall_fn is not None:
            try:
                typed_prompt = _build_typed_prompt(prompt, return_type, schema)
                completion = self.subcall_fn(typed_prompt, model)
                self._pending_llm_calls.append(completion)
                return _parse_typed_response(completion.response, return_type)
            except Exception as e:
                return f"Error: RLM query failed - {e}"

        # Fall back to plain LM call if no recursive capability
        return self._llm_query(prompt, model, return_type=return_type, schema=schema)

    def _rlm_query_batched(self, queries: list[tuple[str, type, dict | None]], model: str | None = None) -> list[Any]:
        """Spawn recursive RLM sub-calls for multiple prompts.

        Each prompt gets its own child RLM for deeper thinking.
        Falls back to llm_query_batched if no recursive capability is configured.

        Args:
            queries: List of (prompt, return_type, schema) tuples.
            model: Optional model name override for the children.

        Returns:
            List of responses in the same order as input queries, each parsed to its return_type.
        """
        if self.subcall_fn is not None:
            results = []
            for prompt, return_type, schema in queries:
                try:
                    typed_prompt = _build_typed_prompt(prompt, return_type, schema)
                    completion = self.subcall_fn(typed_prompt, model)
                    self._pending_llm_calls.append(completion)
                    results.append(_parse_typed_response(completion.response, return_type))
                except Exception as e:
                    results.append(f"Error: RLM query failed - {e}")
            return results

        # Fall back to plain batched LM call if no recursive capability
        return self._llm_query_batched(queries, model)

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: The context data to add
            context_index: Optional explicit index. If None, auto-increments.

        Returns:
            The context index used.
        """
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()")
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)"
            )

        # Alias context_0 as 'context' for backward compatibility
        if context_index == 0:
            self.execute_code(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for a new completion call."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable.

        Args:
            message_history: The list of message dicts from a completion call
            history_index: Optional explicit index. If None, auto-increments.

        Returns:
            The history index used.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Store deep copy to avoid reference issues with nested dicts
        self.locals[var_name] = copy.deepcopy(message_history)

        # Alias history_0 as 'history' for convenience
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    def append_compaction_entry(self, entry: list[dict[str, Any]] | dict[str, Any]) -> None:
        """
        Append a trajectory segment or a summary to the compaction history.

        Entry is either a list of message dicts (trajectory segment) or
        a dict with "type": "summary" and "content": str.
        """
        if not self.compaction:
            return
        self._compaction_history.append(copy.deepcopy(entry))

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory for execution."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def _restore_scaffold(self) -> None:
        """Restore scaffold names after execution so overwrites (e.g. context = 'x') don't persist."""
        for name in RESERVED_TOOL_NAMES:
            if name == "llm_query":
                self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched":
                self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query":
                self.globals["rlm_query"] = self._rlm_query
            elif name == "rlm_query_batched":
                self.globals["rlm_query_batched"] = self._rlm_query_batched
            elif name == "FINAL_VAR":
                self.globals["FINAL_VAR"] = self._final_var
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "context" and "context_0" in self.locals:
                ctx = self.locals["context_0"]
                self.locals["context"] = ctx
                self.globals["context"] = ctx
            elif name == "history" and "history_0" in self.locals and not self.compaction:
                hist = self.locals["history_0"]
                self.locals["history"] = hist
                self.globals["history"] = hist
            elif name == "history" and self.compaction:
                self.locals["history"] = self._compaction_history
                self.globals["history"] = self._compaction_history

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result."""
        start_time = time.perf_counter()

        # Clear pending LLM calls from previous execution
        self._pending_llm_calls = []

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                # Merge user locals into globals so all user-defined names are reachable
                # through __globals__ of any function defined here.  This ensures that
                # after the mock→real function swap, functions look up llm_query (and each
                # other) via self.globals rather than a stale per-call copy.
                self.globals.update(self.locals)
                exec(code, self.globals, self.globals)

                # Sync new/updated user variables back to self.locals
                for key, value in list(self.globals.items()):
                    if not key.startswith("_") and key not in _GLOBALS_ONLY_KEYS:
                        self.locals[key] = value

                # Restore scaffold so model overwrites (context = ..., llm_query = ...) don't persist
                self._restore_scaffold()

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        final_answer = self._last_final_answer
        self._last_final_answer = None

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
            final_answer=final_answer,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up temp directory and reset state."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        if hasattr(self, "globals"):
            self.globals.clear()
        if hasattr(self, "locals"):
            self.locals.clear()

    def __del__(self):
        self.cleanup()
