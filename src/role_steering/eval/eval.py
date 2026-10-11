from typing import List, Dict, Any
import numpy as np
from role_steering.data.dataset import SampleRunnable


GPTOSS_FINAL_MARKER = "<|channel|>final<|message|>"


def extract_final_answer(response: str) -> str:
    # keep only the final channel of a gpt-oss response; empty if the model never got there
    if GPTOSS_FINAL_MARKER not in response:
        return ""
    return response.split(GPTOSS_FINAL_MARKER)[-1].split("<|")[0].strip()


def eval_exact(sample: SampleRunnable, responses: List[str]) -> np.ndarray:
    return np.array([sample.expected_output_value == extract_final_answer(r) for r in responses])


def eval_all(samples: List[SampleRunnable], responses: List[List[str]], cfg: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
    return {"alignment": [eval_exact(s, r) for s, r in zip(samples, responses)]}
