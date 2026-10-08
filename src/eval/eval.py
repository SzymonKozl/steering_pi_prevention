from typing import List, Dict, Any
import numpy as np


def eval_exact(sample: SampleRunnable, responses: List[str]) -> np.ndarray:
    return np.array([sample.expected_output == r for r in responses]) 


def eval_all(samples: List[SampleRunnable], responses: List[List[str]], cfg: Dict[str, Any]) -> List[np.ndarray]:
    return [eval_exact(s, r) for s, r in zip(samples, responses)]
