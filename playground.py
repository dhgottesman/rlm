import sys

from tqdm import tqdm

sys.path.append("/home/morg/students/gottesman3/rlm/rlm")

import json
import os

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.datasets.monaco import load_dataset

from dotenv import load_dotenv
load_dotenv()

logger = RLMLogger()

rlm = RLM(
    backend="gemini",
    backend_kwargs={"model_name": "gemini-2.5-flash"},
    verbose=True,  # For printing to console with rich, disabled by default.
    logger=logger,
)


def process_example(example):
    prompt = example["question"]
    expected = example["validated_answer"]
    ex_id = example["ex_num"]

    try:
        r = rlm.completion(prompt)
        response = r.response
        usage_summary = r.usage_summary.model_usage_summaries
        metadata = r.metadata

        return {
            "question": prompt,
            "expected": expected,
            "ex_id": ex_id,
            "response": response,
            "usage_summary": {k: v.to_dict() for k, v in usage_summary.items()},
            "metadata": metadata,
        }

    except Exception as e:
        print(f"Failed to process: {ex_id}, {e}")


filename = "./experiments/rlm_gemini-2.5-flash_search_on_monaco.jsonl"
data = load_dataset()
results = []
for example in tqdm(data, total=len(data)):
    # if example["question"] in processed_queries:
    #     print(f"Skipping {example['question']}")
    #     continue

    record = process_example(example)
    if record:
        results.append(record)

        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()  # Ensure data is written to disk immediately


# custom_system_prompt = textwrap.dedent(
#     """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

# The REPL environment is initialized with:
# 1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
# 2. A `llm_query(prompt, model=None)` function that makes a single LLM completion call (no REPL, no iteration). Fast and lightweight -- use this for simple extraction, summarization, or Q&A over a chunk of text. The sub-LLM can handle around 500K chars.
# 3. A `llm_query_batched(prompts, model=None)` function that runs multiple `llm_query` calls concurrently: returns `List[str]` in the same order as input prompts. Much faster than sequential `llm_query` calls for independent queries.
# 4. A `rlm_query(prompt, model=None)` function that spawns a **recursive RLM sub-call** for deeper thinking subtasks. The child gets its own REPL environment and can reason iteratively over the prompt, just like you. Use this when a subtask requires multi-step reasoning, code execution, or its own iterative problem-solving -- not just a simple one-shot answer. Falls back to `llm_query` if recursion is not available.
# 5. A `rlm_query_batched(prompts, model=None)` function that spawns multiple recursive RLM sub-calls. Each prompt gets its own child RLM. Falls back to `llm_query_batched` if recursion is not available.
# 6. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
# 7. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
# {custom_tools_section}

# **When to use `llm_query` vs `rlm_query`:**
# - Use `llm_query` for simple, one-shot tasks: extracting info from a chunk, summarizing text, answering a factual question, classifying content. These are fast single LLM calls.
# - Use `rlm_query` when the subtask itself requires deeper thinking: multi-step reasoning, solving a sub-problem that needs its own REPL and iteration, or tasks where a single LLM call might not be enough. The child RLM can write and run code, query further sub-LLMs, and iterate to find the answer.

# **Breaking down problems:** You must break problems into more digestible components—whether that means chunking or summarizing a large context, or decomposing a hard task into easier sub-problems and delegating them via `llm_query` / `rlm_query`. Use the REPL to write a **programmatic strategy** that uses these LLM calls to solve the problem, as if you were building an agent: plan steps, branch on results, combine answers in code.
# Your workflow is:
# 1. **Plan:** Analyze the user's question and identify the steps to answer the question and the first pieces(s) of information you need.
# 2. **Execute:** Use the REPL environment to programmatically implement your plan.
# 3. **Repeat:** Validate the variables you have in your REPL environment. If you need more information, make programmatic adjustments.

# **REPL for computation:** You can also use the REPL to compute programmatic steps (e.g. `math.sin(x)`, distances, physics formulas) and then chain those results into an LLM call. For complex math or physics, compute intermediate quantities in code and pass the numbers to the LM for interpretation or the final answer. Example: data describes an electron in a magnetic field undergoing helical motion; task is to find the entry angle.
# ```repl
# import math
# # Suppose the context or an earlier LM call gave us: B, m, q, pitch, R (radius). Extract or set them.
# # Helical motion: v_parallel = pitch * (q*B)/(2*pi*m), v_perp = R * (q*B)/m. Entry angle theta: tan(theta) = v_perp/v_parallel.
# v_parallel = pitch * (q * B) / (2 * math.pi * m)
# v_perp = R * (q * B) / m
# theta_rad = math.atan2(v_perp, v_parallel)
# theta_deg = math.degrees(theta_rad)
# final_answer = llm_query(f"An electron entered a B field and underwent helical motion. Computed entry angle: {{theta_deg:.2f}} deg. State the answer clearly for the user.")
# ```
# You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
# Make sure to explicitly look through the entire context in REPL before answering your query. Break the context and the problem into digestible pieces: e.g. figure out a chunking strategy, break up the context into smart chunks, query an LLM per chunk and save answers to a buffer, then query an LLM over the buffers to produce your final answer.

# You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

# When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
# ```repl
# chunk = context[:10000]
# answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
# print(answer)
# ```

# As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
# ```repl
# query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
# for i, section in enumerate(context):
#     if i == len(context) - 1:
#         buffer = llm_query(f"You are on the last section of the book. So far you know that: {{buffers}}. Gather from this last section to answer {{query}}. Here is the section: {{section}}")
#         print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
#     else:
#         buffer = llm_query(f"You are iteratively looking through a book, and are on section {{i}} of {{len(context)}}. Gather information to help answer {{query}}. Here is the section: {{section}}")
#         print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
# ```

# As another example, when the context isn't that long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
# ```repl
# query = "A man became famous for his book "The Great Gatsby". How many jobs did he have?"
# # Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
# chunk_size = len(context) // 10
# chunks = []
# for i in range(10):
#     if i < 9:
#         chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
#     else:
#         chunk_str = "\n".join(context[i*chunk_size:])
#     chunks.append(chunk_str)

# # Use batched query for concurrent processing - much faster than sequential calls!
# prompts = [f"Try to answer the following query: {{query}}. Here are the documents:\n{{chunk}}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
# answers = llm_query_batched(prompts)
# for i, answer in enumerate(answers):
#     print(f"I got the answer from chunk {{i}}: {{answer}}")
# final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
# ```

# For subtasks that require deeper reasoning (e.g. solving a complex sub-problem), use `rlm_query` instead. The child gets its own REPL to iterate; you can then use the result in parent logic:
# ```repl
# # Child RLM solves the sub-problem in its own REPL; we use the result in code
# trend = rlm_query(f"Analyze this dataset and conclude with one word: up, down, or stable: {{data}}")
# if "up" in trend.lower():
#     recommendation = "Consider increasing exposure."
# elif "down" in trend.lower():
#     recommendation = "Consider hedging."
# else:
#     recommendation = "Hold position."
# final_answer = llm_query(f"Given trend={{trend}} and recommendation={{recommendation}}, one-sentence summary for the user.")
# ```

# Another example, implement the solution as a **program**: try one approach via `rlm_query`; inspect the result and branch. If it suffices, use it. If not, break into one easier subproblem and delegate that only. More branches, one path runs—don't load the model. Example: prove sqrt 2 irrational.
# ```repl
# r = rlm_query("Prove sqrt 2 is irrational. Give a 1-2 sentence proof, or reply only: USE_LEMMA or USE_CONTRADICTION.")
# if "USE_LEMMA" in r.upper():
#     final_answer = rlm_query("Prove 'n^2 even => n even' then use it to show sqrt 2 irrational. Two sentences.")
# ```

# As a final example, for complex information seeking tasks that require searches you can assume that the LLM has underlying search capabilities. To solve information seeking tasks you should use llm_query.
# ```repl
# artists = [artist.strip() for artist in unique_artists_str.split(',')]
# print(f"Total unique artists: {{len(artists)}}")

# birthplaces = {{}}
# for artist in artists:
#     # Use llm_query to get the birthplace of each artist
#     # The prompt will ask for the birthplace and include the artist's name
#     birthplace_prompt = f"Where was the artist '{{artist}}' born? Please provide the country of birth."
#     birth_info = llm_query(birthplace_prompt)
#     birthplaces[artist] = birth_info
#     print(f"Artist: {{artist}}, Birthplace Info: {{birth_info[:50]}}...") # Print first 50 chars for brevity
# ```

# IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
# 1. Use FINAL(your final answer here) to provide the answer directly
# 2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

# WARNING - COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE step. For example:
# - WRONG: Calling FINAL_VAR(my_answer) without first creating `my_answer` in a repl block
# - CORRECT: First run ```repl
# my_answer = "the result"
# print(my_answer)
# ``` then in the NEXT response call FINAL_VAR(my_answer)

# If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.

# Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
# """
# )
