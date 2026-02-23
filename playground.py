import sys
sys.path.append("/home/morg/students/gottesman3/rlm/rlm")

import os
from rlm import RLM


rlm = RLM(
    backend="qwen",
    backend_kwargs={"model_name": "Qwen/Qwen3-4B"},
    verbose=True,  # For printing to console with rich, disabled by default.
)

print(rlm.completion("What percentage of artists in the Billboard Top 100 in 2020 were born outside of the United States?").response)