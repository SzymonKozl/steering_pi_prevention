from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class SampleRunnable:
    messages: List[Dict[str, str]]
    repeats: int
    temperature: float
    desired_behaviour: Optional[str]
    expected_output_value: Optional[str]
    name: str


def dispatch_examples(config: Dict[str, Any]) -> List[SampleRunnable]:
    """
    Args:
        config: Evaluation configuration dictionary.
    """
    res = []
    for set_name, set_cfg in config["samples"].items():
        path = Path(set_cfg["path"])
        for file in sorted(path.glob("*.json")) if path.is_dir() else [path]:
            with open(file, "r", encoding="utf-8") as f:
                examples = json.load(f)
            for name, ex in examples.items():
                res.append(SampleRunnable(
                    messages=ex["messages"],
                    repeats=config["repeats_per_example"],
                    temperature=config["temperature"],
                    desired_behaviour=ex.get("desired_behaviour"),
                    expected_output_value=ex.get("expected_output"),
                    name=f"{set_name}/{name}",
                ))
    return res
