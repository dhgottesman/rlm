import json
import os

from huggingface_hub import hf_hub_download


def load_dataset():
    file_path = hf_hub_download(
        repo_id="allenai/MoNaCo_Benchmark",
        filename="monaco_version_1_release.jsonl",
        token=os.environ.get("HF_TOKEN"),
        repo_type="dataset",
    )

    print(f"File downloaded to: {file_path}")

    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data
