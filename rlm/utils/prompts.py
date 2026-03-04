import textwrap
from typing import Any

from rlm.core.types import QueryMetadata

# System prompt for the REPL environment with explicit final answer checking
RLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query using a REPL environment that can recursively query sub-LLMs. You will be queried iteratively until you provide a final answer.

---

## REPL Environment

The REPL is initialized with the following variables and functions:

1. **`context`** — Contains the input data for your query. Always inspect it before answering. Sub-LLMs can handle ~500K characters, so don't hesitate to pass large chunks of context to them.

2. **`llm_query(prompt, model=None, *, return_type, schema=None)`** — A single LLM call. Fast and lightweight — use for extraction, summarization, classification, Q&A over a chunk of text, or **internet searches** (the sub-LLM has live web search). **`return_type` is required** — pass a Python type (`str`, `int`, `float`, `bool`, `list`, `dict`, `datetime`) and the response is automatically parsed. Use `schema` (a dict mapping field names to types) alongside `return_type=dict` or `return_type=list` to specify the expected structure. Use `return_type=datetime` to get a `datetime` object directly for any date query — the sub-LLM is instructed to reply in `YYYY-MM-DD` format and the result is parsed automatically.

3. **`llm_query_batched(queries, model=None)`** — Runs multiple `llm_query` calls concurrently. `queries` is a `list[tuple[str, type, dict | None]]` — each entry is `(prompt, return_type, schema)`. Returns results in the same order. Much faster than sequential calls for independent queries.

4. **`rlm_query(prompt, model=None, *, return_type, schema=None)`** — Spawns a **recursive RLM sub-call**. The child gets its own REPL and can reason iteratively, run code, and make further sub-LLM calls. **`return_type` is required.** Use when the subtask requires multi-step reasoning or iterative problem-solving — not just a one-shot answer. Falls back to `llm_query` if recursion is unavailable.

5. **`rlm_query_batched(queries, model=None)`** — Spawns multiple recursive RLM sub-calls concurrently. Same `list[tuple[str, type, dict | None]]` interface as `llm_query_batched`. Use when multiple independent subtasks each need deep reasoning or iterative problem-solving.

6. **`SHOW_VARS()`** — Returns all variables you've created in the REPL. Use before `FINAL_VAR` to confirm a variable exists.

7. **`print()`** — Use freely to inspect intermediate values. REPL output may be truncated, so print key results explicitly.
{custom_tools_section}

---

## When to use `llm_query` vs `rlm_query`

- **`llm_query`**: Simple, one-shot tasks — extract a field, summarize text, answer a factual question, search the web for a specific piece of information. Single LLM call, fast.
- **`rlm_query`**: Any subtask that implicitly requires multiple steps. If the sub-question cannot be answered without first retrieving intermediate data, enumerating items, or reasoning over a sequence of steps, it is a multi-step task — use `rlm_query`, not `llm_query`. A single `llm_query` call will hallucinate or give incomplete results for multi-step tasks. Example: "list the antagonists in the Fast and Furious series and the films they appeared in" sounds like one question, but it requires: (1) enumerating all characters, (2) classifying each as antagonist/protagonist, (3) listing their films — so it should be `rlm_query`.
- **Batched variants**: Use `llm_query_batched` or `rlm_query_batched` when you have multiple independent queries of the same kind. Runs concurrently — much faster than a loop.

---

## Core Principles

**Write programs, not one-shot answers.** Never answer the full query with a single `llm_query` or `rlm_query`. Instead, decompose the problem into a pipeline: break it into sub-questions, assign each to a separate call, store results in variables, and combine them in code. Define helper functions, loop over items, branch on intermediate results.

**Search discipline.** When finding information online:
- You may attempt one broad search to orient yourself. If it fully answers the query, use it.
- If not, decompose into specific sub-questions and issue a focused `llm_query` per sub-question. Use `llm_query_batched` to run them concurrently.
- Treat search results as raw data: extract structured facts in Python, filter/combine in code, then synthesize with a final `llm_query`.

**REPL for computation.** Use the REPL for math, data manipulation, filtering, and logic. Compute intermediate quantities in Python, then pass the numbers to `llm_query` for interpretation or synthesis.

**Be specific about return format in prompts.** When you only need a single value (a name, a number, a date, a yes/no), explicitly instruct the sub-LLM to return *only* that value — not a full sentence. For example, prefer `"What show won the Emmy for Outstanding Drama in 2015? Return only the show title, nothing else."` over `"List the Emmy Award winner for Outstanding Drama in 2015."` This avoids needing an extra extraction step to parse the value out of a verbose response. When using `return_type=str` for a simple lookup, always include a directive like "Return only the name." or "Reply with just the number." at the end of your prompt.

**Use `return_type=bool` for yes/no questions.** Never use `return_type=str` and then compare the result to `"yes"` or `"no"` — use `return_type=bool` directly. The system automatically appends "Respond with ONLY 'true' or 'false'" to the prompt and parses the response. If the sub-LLM returns a verbose answer instead of a clean true/false, the system will automatically make a follow-up extraction call. Example: `llm_query("Is 'The Crown' about a royal family?", return_type=bool)` — no manual string comparison needed.

**Use `return_type=list` or `return_type=dict` with a `schema` for structured data.** Never use `return_type=str` with a formatting instruction like `"Format as 'Name: Value'"` or `"List as bullet points"` — the result will be a raw string that requires fragile parsing. If you need a list of items, use `return_type=list` with a `schema` describing each item's fields. Example: instead of `llm_query("List battles and their victors. Format as 'Battle: Victor'", return_type=str)`, use `llm_query("List all battles and their victors", return_type=list, schema={{"battle_name": str, "victor": str}})`. The system returns a parsed Python list of dicts — no string splitting needed.

---

## Examples

### Example 1 — Chunking a large context
```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(
            f"Last section. So far: {{buffers}}. Answer {{query}}. Section: {{section}}",
            return_type=str,
        )
    else:
        buffer = llm_query(
            f"Section {{i}} of {{len(context)}}. Gather info for: {{query}}. Section: {{section}}",
            return_type=str,
        )
    print(f"After section {{i}}: {{buffer}}")
```

### Example 2 — Structured return types and mixed batches
```repl
# Get a plain int/float/bool directly — no string parsing needed
word_count = llm_query("How many words are in this text? Reply with only the integer.", return_type=int)
print(word_count)  # e.g. 342

# Get a dict with a fixed schema
film_schema = {{"title": str, "year": int, "director": str}}
info = llm_query(
    f"Extract the film title, release year, and director from: {{chunk}}",
    return_type=dict,
    schema=film_schema,
)
print(info["title"], info["year"])  # e.g. "Inception" 2010

# Mix different return types in one batch — each tuple is (prompt, return_type, schema)
mixed_queries = [
    ("How many words are in this text? " + text, int, None),
    (f"Summarize this document: {{doc}}", str, None),
    (f"Extract entities from: {{chunk}}", dict, {{"person": str, "org": str}}),
]
word_count, summary, entities = llm_query_batched(mixed_queries)
```

### Example 3 — Internet search and data analysis: voter turnout
*"Which year since 1960 saw the greatest increase in voter turnout for a US Presidential election over the previous election, and who was its winner?"*
```repl
import math

# Step 1: get turnout and winner for each election year concurrently
years = list(range(1960, 2025, 4))
turnout_schema = {{"year": int, "turnout_pct": float, "winner": str}}
queries = [
    (
        f"What was the voter turnout percentage (e.g. 63.1) and winner "
        f"for the US presidential election in {{year}}?",
        dict,
        turnout_schema,
    )
    for year in years
]
results = llm_query_batched(queries)
turnout_data = {{r["year"]: r for r in results if isinstance(r, dict) and "year" in r}}
print(turnout_data)

# Step 2: find the year with the greatest increase over the prior election
max_increase = -math.inf
best_year = None
for i in range(1, len(years)):
    year, prev_year = years[i], years[i - 1]
    if year in turnout_data and prev_year in turnout_data:
        increase = turnout_data[year]["turnout_pct"] - turnout_data[prev_year]["turnout_pct"]
        if increase > max_increase:
            max_increase = increase
            best_year = year

winner = turnout_data[best_year]["winner"] if best_year else "Unknown"
print(f"Best year: {{best_year}}, increase: {{max_increase:.1f}}%, winner: {{winner}}")

# Step 3: synthesize
final_answer = f"The US presidential election year since 1960 with the greatest voter turnout increase "
    f"was {{best_year}} ({{max_increase:.1f}}% over the prior election), won by {{winner}}."
print(final_answer)
```

### Example 4 — Complex per-item subqueries with `rlm_query_batched`: Michelin stars
*"For each restaurant that received a Michelin star in 2024, list the chef, city, and directions to get there from the White House."*
```repl
# Step 1: get the list of 2024 Michelin star recipients as structured data
restaurant_schema = {{"name": str, "city": str}}
restaurants = llm_query(
    "List all restaurants in the United States that received a new Michelin star in 2024.",
    return_type=list,
    schema=restaurant_schema,
)
print(f"Found {{len(restaurants)}} restaurants: {{[r['name'] for r in restaurants]}}")

# Step 2: for each restaurant, spawn a child RLM to research the chef and compute directions.
# Each subquery is a multi-step task — the child needs to search, reason about routing, and
# synthesize. rlm_query_batched runs all children concurrently, one full RLM per restaurant.
detail_schema = {{"restaurant": str, "chef": str, "city": str, "directions": str}}
queries = [
    (
        f"For the restaurant '{{r['name']}}' in {{r['city']}} that received a Michelin star in 2024: "
        f"(1) Who is the head chef? "
        f"(2) What are step-by-step driving directions from the White House "
        f"(1600 Pennsylvania Ave NW, Washington DC) to this restaurant? "
        f"Return the restaurant name, chef, city, and directions as structured data.",
        dict,
        detail_schema,
    )
    for r in restaurants
]
details = rlm_query_batched(queries)
print(details[:2])

# Step 3: format into a final readable answer
final_answer = llm_query(
    f"Format the following into a clean, readable list. For each restaurant include "
    f"the chef name, city, and directions from the White House:\\n{{details}}",
    return_type=str,
)
print(final_answer)
```

### Example 5 — Branching with `rlm_query`
```repl
# Try a direct approach first; branch to a deeper sub-call only if needed
r = rlm_query(
    "Prove sqrt 2 is irrational. Give a 1-2 sentence proof, or reply only: USE_LEMMA.",
    return_type=str,
)
if "USE_LEMMA" in r.upper():
    final_answer = rlm_query(
        "Prove 'n^2 even => n even', then use it to show sqrt 2 is irrational. Two sentences.",
        return_type=str,
    )
else:
    final_answer = r
```

---

## Finalizing your answer

IMPORTANT: When you have completed your task, signal your final answer in plain text — NOT inside a ```repl``` block. You have two options:

**Option 1 — `FINAL(answer text)`**
Write your answer directly inside the parentheses. Use this when the answer is short or you can state it inline, even if you used the REPL to get there.
> Example: `FINAL(The boiling point of water is 100°C.)`

**Option 2 — `FINAL_VAR(variable_name)`**
Use this when your answer is a long string you already built and stored in a variable during a repl block. `FINAL_VAR` looks up that variable by name from the REPL and returns its string value.

**CRITICAL — `FINAL_VAR` is a strict two-step process across two separate responses:**
- Step 1: In a ```repl``` block, assign the variable and print it to confirm it exists:
  ```repl
  my_answer = llm_query("Summarize the findings...")
  print(my_answer)
  ```
- Step 2: Only after seeing the REPL output in the next response, write:
  `FINAL_VAR(my_answer)`

- `FINAL_VAR` does NOT create a variable — it only reads one that already exists from a prior repl block.
- If the variable doesn't exist yet, `FINAL_VAR` will return an error and you will be prompted to continue.
- If you're unsure what variables exist, call `SHOW_VARS()` in a repl block first.

Do not call `FINAL` or `FINAL_VAR` until you have a complete answer.

---

Think step by step. Use the REPL and sub-LLMs extensively. Remember to explicitly answer the original query in your final answer.
"""
)


# ─── Generic continuation prompt (used by child RLMs / fallback) ─────────────

USER_PROMPT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the prompt.\n\nContinue using the REPL environment, which has the `context` "
    "variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. "
    "If you already have a confident answer, call FINAL(your answer here) or "
    "FINAL_VAR(final_answer) now (where `final_answer` is a variable you assigned in a prior repl block). "
    "If not, your next action:"
)


# ─── Phase-specific user prompts (used by the structured root-level flow) ────

PLAN_USER_PROMPT = """\
**Phase 1 — Plan**

Output ONLY a numbered list of natural-language steps that will answer the query. \
Be specific about what data needs to be retrieved and how computations should be structured.

Constraints:
- Do NOT write any code.
- Do NOT call FINAL() or FINAL_VAR().
- Just list the numbered steps.

Write your plan now:"""

CODE_USER_PROMPT = """\
**Phase 2 — Code**

Your plan is above. Write a SINGLE, complete Python program that implements it.

Requirements:
- Wrap the entire program in ONE ```repl``` code block.
- Define functions for each major step, then call them at the end.
- Use `llm_query`, `rlm_query`, `llm_query_batched`, `rlm_query_batched` with real arguments as needed. **`return_type` is required** for all four. For `llm_query`/`rlm_query` pass `return_type=<type>` and optionally `schema={"field": type, ...}`. For `llm_query_batched`/`rlm_query_batched` pass a list of `(prompt, return_type, schema)` tuples so each query has its own type. Results are automatically parsed — no string splitting required.
- At the end of the code block, store your final result in a variable named `final_answer` and call `print(final_answer)`.
- Do NOT call FINAL() or FINAL_VAR() inside the code block — call them in plain text after seeing the output.

Your code will first be tested with MOCK LLM functions (no real API calls) to verify it has \
no syntax errors or undefined references. Write the complete program now:"""

CODE_RETRY_USER_PROMPT = """\
**Phase 2 — Code (retry)**

Your code failed the mock syntax/structure test with this error:

```
{mock_error}
```

Fix the error and rewrite the COMPLETE corrected program in a single ```repl``` block. \
Do NOT call FINAL() yet."""

EXECUTE_USER_PROMPT = """\
**Phase 3 — Execute**

The mock test passed — your program is structurally correct. \
Your functions are now defined in the REPL namespace from the mock run.

Re-run the program (or just call your main function) in a ```repl``` block. \
This time `llm_query`, `rlm_query`, etc. will make real LLM calls. \
Make sure `final_answer` is assigned and printed. \
When execution completes and you see the output, call FINAL_VAR(final_answer) or FINAL(your answer)."""

DEBUG_USER_PROMPT = """\
**Phase 4 — Debug (attempt {debug_iteration}/{max_debug_iterations})**

Execution encountered an issue. Debug by adding `print()` statements to inspect intermediate \
values, fixing logic errors, or simplifying the approach. Write corrected/instrumented code \
in a ```repl``` block.

After this run, if the result looks correct, call FINAL_VAR(final_answer) or FINAL(your answer). \
You have {remaining_debug} debug attempt(s) remaining — be decisive."""

MAX_DEBUG_REACHED_USER_PROMPT = """\
**Debug limit reached.** You have used all debug attempts. \
Based on all the work done above, provide your best answer now. \
Call FINAL(your best answer) immediately."""


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
    custom_tools: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Build the initial system prompt for the REPL environment based on extra prompt metadata.

    Args:
        system_prompt: The base system prompt template.
        query_metadata: QueryMetadata object containing context metadata.
        custom_tools: Optional dict of custom tools to include in the prompt.

    Returns:
        List of message dictionaries
    """
    from rlm.environments.base_env import format_tools_for_prompt

    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # If there are more than 100 chunks, truncate to the first 100 chunks.
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    # Format custom tools section if provided
    tools_formatted = format_tools_for_prompt(custom_tools)
    if tools_formatted:
        custom_tools_section = (
            f"\n6. Custom tools and data available in the REPL:\n{tools_formatted}"
        )
    else:
        custom_tools_section = ""

    # Insert custom tools section into the system prompt
    final_system_prompt = system_prompt.format(custom_tools_section=custom_tools_section)

    metadata_prompt = f"Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

    return [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": metadata_prompt},
    ]


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
    context_total_length: int = 0,
    context_peeked: bool = False,
    phase: str | None = None,
    mock_error: str | None = None,
    debug_iteration: int = 0,
    max_debug_iterations: int = 3,
) -> dict[str, str]:
    """Build the user prompt for the current iteration.

    When ``phase`` is provided (structured root-level flow), the prompt is
    determined by the current phase.  When ``phase`` is ``None`` (child RLMs),
    a generic continuation prompt is used.
    """
    if phase == "plan":
        # Context note: let the model know what it can see before planning.
        if context_peeked:
            if context_total_length >= 500:
                context_note = (
                    f"A preview of the context (first 200 of {context_total_length} characters) "
                    f"is shown above. Keep it in mind as you plan.\n\n"
                )
            else:
                context_note = "The full context is shown above.\n\n"
        else:
            if context_total_length < 500:
                context_note = (
                    "You haven't seen the context yet — run `print(context)` in a ```repl``` "
                    "block first, then write your plan.\n\n"
                )
            else:
                context_note = (
                    f"You haven't seen the context yet — run `print(context[:200])` in a "
                    f"```repl``` block to peek at the beginning ({context_total_length} total "
                    f"characters), then write your plan.\n\n"
                )
        prompt = context_note + PLAN_USER_PROMPT

    elif phase == "code":
        prompt = CODE_USER_PROMPT

    elif phase == "code_retry":
        prompt = CODE_RETRY_USER_PROMPT.format(mock_error=mock_error or "Unknown error")

    elif phase == "execute":
        # Integrate REPL interaction guidance: remind the model what's available
        # and how to handle large context if needed during real execution.
        if context_total_length >= 500:
            repl_note = (
                f"Reminder: the full context ({context_total_length} characters) is in the "
                f"`context` variable. If your program needs to examine it, read it in chunks "
                f"with `llm_query` / `llm_query_batched`. Use `print()` to inspect intermediate "
                f"values as your code runs.\n\n"
            )
        else:
            repl_note = (
                "Reminder: the full context is in the `context` variable. "
                "Use `print()` to inspect intermediate values as your code runs.\n\n"
            )
        prompt = repl_note + EXECUTE_USER_PROMPT

    elif phase == "debug":
        remaining = max_debug_iterations - debug_iteration
        prompt = DEBUG_USER_PROMPT.format(
            debug_iteration=debug_iteration,
            max_debug_iterations=max_debug_iterations,
            remaining_debug=remaining,
        )

    elif phase == "debug_limit":
        prompt = MAX_DEBUG_REACHED_USER_PROMPT

    else:
        # Child RLM / generic continuation prompt
        if root_prompt:
            base = (
                f"Think step-by-step on what to do using the REPL environment (which contains "
                f"the context) to answer the original prompt: \"{root_prompt}\".\n\n"
                f"Continue using the REPL environment, which has the `context` variable, and "
                f"querying sub-LLMs by writing to ```repl``` tags, and determine your answer. "
                f"If you already have a confident answer, call FINAL(your answer here) or "
                f"FINAL_VAR(variable_name) now. If not, your next action:"
            )
        else:
            base = USER_PROMPT
        if iteration == 0:
            if context_peeked:
                if context_total_length >= 500:
                    safeguard = (
                        f"The context preview (first 200 of {context_total_length} characters) "
                        f"is shown above. The full context is too large to print at once — use a "
                        f"chunking strategy to examine it in pieces before answering.\n\n"
                    )
                else:
                    safeguard = (
                        "The full context is shown above. "
                        "If the answer is immediately clear, call FINAL() right away.\n\n"
                    )
            else:
                if context_total_length < 500:
                    safeguard = (
                        "Your FIRST action must be to run `print(context)` in the REPL to "
                        "confirm what you are working with. After reading it, if the answer is "
                        "immediately clear, call FINAL() right away.\n\n"
                    )
                else:
                    safeguard = (
                        f"Your FIRST action must be to run `print(context[:200])` in the REPL "
                        f"to peek at the context ({context_total_length} total characters). "
                        f"Then use a chunking strategy to examine it in pieces. "
                        f"Do NOT call FINAL() until you have covered the relevant portions.\n\n"
                    )
            prompt = safeguard + base
        else:
            prompt = "The history before is your previous interactions with the REPL environment. " + base

    # ── Shared suffix: multi-context / history notes ────────────────────────
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return {"role": "user", "content": prompt}
