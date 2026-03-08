import json
import os
import sys
import time

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append("/home/morg/students/gottesman3/rlm/rlm")

from rlm import RLM
from rlm.datasets.monaco import load_dataset
from rlm.logger import RLMLogger
from google.genai import errors as genai_errors


load_dotenv()

logger = RLMLogger()

rlm = RLM(
    backend="gemini",
    backend_kwargs={"model_name": "gemini-2.5-flash"},
    verbose=True,
    logger=logger,
)


def process_example(example):
    prompt = example["question"]
    expected = example["validated_answer"]
    ex_id = example["ex_num"]

    r = rlm.completion(prompt)
    return {
        "question": prompt,
        "expected": expected,
        "ex_id": ex_id,
        "response": r.response,
        "usage_summary": {k: v.to_dict() for k, v in r.usage_summary.model_usage_summaries.items()},
        "metadata": r.metadata,
    }


filename = "/home/morg/students/gottesman3/rlm/experiments/rlm_gemini-2.5-flash_monaco_v2.jsonl"
data = load_dataset()
results = []

processed_queries = set()
if os.path.exists(filename):
    with open(filename, encoding="utf-8") as f:
        for line in f:
            processed_queries.add(json.loads(line)["question"])

total_tokens = 0

for example in tqdm(data, total=len(data)):
    if example["question"] in processed_queries:
        print(f"Skipping {example['question']}")
        continue

    while True:
        try:
            record = process_example(example)
            break
        except genai_errors.ClientError as e:
            if e.code == 429:
                print(f'Rate limited, retrying: {repr(e)}')
                time.sleep(61)
            else:
                raise  # don't retry other 4xx errors
        except genai_errors.ServerError as e:
            if e.code == 503:
                print(f'Service unavailable, retrying: {repr(e)}')
                time.sleep(61)
            else:
                raise

    if record:
        results.append(record)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        record_total_tokens = record["usage_summary"]["gemini-2.5-flash"]["total_output_tokens"] + record["usage_summary"]["gemini-2.5-flash"]["total_input_tokens"]

        if total_tokens + record_total_tokens > 1000000:
            time.sleep(61)
            total_tokens = 0