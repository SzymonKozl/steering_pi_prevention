from typing import List, Dict, Optional


class SampleRunnable:
    def __init__(
        self,
        messages: List[Dict[str, str]],
        repeats: int,
        temperature: float,
        desired_behaviour: Optional[str]):
        self.messages = messages
        self.repeats = repeats
        self.temperature = temperature
        self.desired_behaviour = desired_behaviour



def dispatch_examples(config: Dict) -> List[SampleRunnable]:
    # todo
    pass
