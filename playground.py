import json
import os
import sys

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append("/home/morg/students/gottesman3/rlm/rlm")

from rlm import RLM
from rlm.datasets.monaco import load_dataset
from rlm.logger import RLMLogger

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

    try:
        r = rlm.completion(prompt)
        return {
            "question": prompt,
            "expected": expected,
            "ex_id": ex_id,
            "response": r.response,
            "usage_summary": {k: v.to_dict() for k, v in r.usage_summary.model_usage_summaries.items()},
            "metadata": r.metadata,
        }
    except Exception as e:
        print(f"Failed to process: {ex_id}, {e}")


filename = "./experiments/rlm_gemini-2.5-flash_search_on_monaco.jsonl"
data = load_dataset()
results = []

processed_queries = set()
if os.path.exists(filename):
    with open(filename, encoding="utf-8") as f:
        for line in f:
            processed_queries.add(json.loads(line)["question"])

for example in tqdm(data, total=len(data)):
    if example["question"] in processed_queries:
        print(f"Skipping {example['question']}")
        continue

    record = process_example(example)
    if record:
        results.append(record)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
