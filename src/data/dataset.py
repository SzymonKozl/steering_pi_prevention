from typing import Any, Dict, List, Optional


class SampleRunnable:
    def __init__(
        self,
        messages: List[Dict[str, str]],
        repeats: int,
        temperature: float,
        desired_behaviour: Optional[str],
    ):
        """
        Args:
            messages: List of chat messages with role and text.
            repeats: Number of evaluation repeats.
            temperature: Sampling temperature.
            desired_behaviour: Expected model response description.
        """
        self.messages = messages
        self.repeats = repeats
        self.temperature = temperature
        self.desired_behaviour = desired_behaviour


def dispatch_examples(config: Dict[str, Any]) -> List[SampleRunnable]:
    """
    Args:
        config: Evaluation configuration dictionary.
    """
    pass
