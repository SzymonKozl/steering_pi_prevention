from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Optional

import modal
import numpy as np
if modal.is_local():
    import requests

from role_steering.data.dataset import SampleRunnable
from role_steering.data.generation import call_llm_api


logger = logging.getLogger(__name__)


DEFAULT_JUDGE_MODEL = "google/gemini-2.5-pro"


def judge_answer(instruction: str, desired_behaviour: str, model_response: str, model: str = DEFAULT_JUDGE_MODEL) -> Optional[bool]:
    """Returns True/False for yes/no, None if the judge says unclear (or answers off-format)."""
    prompt = instruction.format(answer=model_response, desired_behaviour=desired_behaviour)
    try:
        response = call_llm_api([{"role": "user", "content": prompt}], model=model)
    except requests.RequestException as e:
        # http errors are already logged with details by call_llm_api
        logger.warning(f"judge call failed, scoring as unclear: {e!r}")
        return None
    # content can be null (e.g. safety-filtered by the judge provider) -> treat as unclear
    verdict = (response or "").strip().lower().strip(".\"'` ")
    if verdict.startswith("yes"):
        return True
    if verdict.startswith("no"):
        return False
    return None


def judge_all(samples: List[SampleRunnable], responses: List[List[str]], instruction: str, model: str = DEFAULT_JUDGE_MODEL, max_workers: int = 16) -> List[np.ndarray]:
    """Per sample, an array over repeats: 1.0 yes, 0.0 no, nan unclear."""
    jobs = [(s.desired_behaviour, r) for s, resps in zip(samples, responses) for r in resps]
    with ThreadPoolExecutor(max_workers) as pool:
        verdicts = list(pool.map(lambda j: judge_answer(instruction, *j, model=model), jobs))
    scores = np.array([np.nan if v is None else float(v) for v in verdicts])
    return np.split(scores, np.cumsum([len(r) for r in responses])[:-1])
