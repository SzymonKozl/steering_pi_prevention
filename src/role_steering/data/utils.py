from typing import Dict, List, Optional, Tuple
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def fetch_samples_from_dataset(path: str, sample_no: int, name: Optional[str] = None, split: str = "train", seed: int = 42) -> List[str]:
    ds = load_dataset(path, name, split=split, streaming=True).shuffle(seed=seed, buffer_size=50_000)
    return [sample["text"] for sample in ds.take(sample_no)]


def _to_chat_message(role: str, text: str) -> List[Dict]:
    if role == "cot":
        return [{"role": "assistant", "thinking": text, "content": ""}]
    if role == "tool":
        tool_call = {"type": "function", "function": {"name": "tool", "arguments": {}}}
        return [{"role": "assistant", "tool_calls": [tool_call]}, {"role": "tool", "content": text}]
    return [{"role": role, "content": text}]


_GPTOSS_HEADERS = {
    "system": "system<|message|>",
    "developer": "developer<|message|>",
    "user": "user<|message|>",
    "cot": "assistant<|channel|>analysis<|message|>",
    "assistant": "assistant<|channel|>final<|message|>",
    "tool": "functions. to=assistant<|channel|>commentary<|message|>",
}


def _gptoss_default_system_message(tokenizer: AutoTokenizer) -> str:
    # stock system message (identity, cutoff, date, reasoning effort, channels) as rendered by the chat template
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=False)
    return rendered[:rendered.index("<|start|>user")]


def _tokenize_gptoss(
    input: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool,
    max_tokens_per_message: Optional[int],
    default_system_prompt: bool,
) -> Tuple[torch.Tensor, List[Optional[str]]]:
    segments = []
    if default_system_prompt:
        segments.append((_gptoss_default_system_message(tokenizer), None))
    for m in input:
        if default_system_prompt and m["role"] == "system":
            # mirror the chat template: custom system text goes into a developer message
            segments += [("<|start|>developer<|message|># Instructions\n\n", None), (m["text"], m["role"]), ("\n\n<|end|>", None)]
            continue
        segments += [(f"<|start|>{_GPTOSS_HEADERS[m['role']]}", None), (m["text"], m["role"]), ("<|end|>", None)]
    if add_generation_prompt:
        segments.append(("<|start|>assistant", None))
    input_ids, token_roles = [], []
    for text, role in segments:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if role is not None and max_tokens_per_message is not None:
            ids = ids[:max_tokens_per_message]
        input_ids += ids
        token_roles += [role] * len(ids)
    return torch.tensor([input_ids]), token_roles


def tokenize_with_role_preservation(
    input: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool = False,
    max_tokens_per_message: Optional[int] = None,
    default_system_prompt: bool = False,
) -> Tuple[torch.Tensor, List[Optional[str]]]:
    """
    returns tokenized input (1, T) as well as a list with role for each token (template tokens get None)
    default_system_prompt: (gpt-oss only) prepend the stock system message and render system messages as
        developer instructions, like tokenizer.apply_chat_template does; otherwise messages are rendered bare
    """
    if "gpt-oss" in tokenizer.name_or_path:
        return _tokenize_gptoss(input, tokenizer, add_generation_prompt, max_tokens_per_message, default_system_prompt)
    texts = [m["text"] for m in input]
    if max_tokens_per_message is not None:
        texts = tokenizer.batch_decode(
            tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_tokens_per_message).input_ids
        )
    chat = [cm for m, t in zip(input, texts) for cm in _to_chat_message(m["role"], t)]
    rendered = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=add_generation_prompt)

    spans = []
    cursor = 0
    for m, t in zip(input, texts):
        for candidate in (t, json.dumps(t)[1:-1]):
            start = rendered.find(candidate, cursor)
            if start != -1:
                break
        if start == -1:
            raise ValueError(f"message with role {m['role']} not found in the rendered chat template")
        cursor = start + len(candidate)
        spans.append((start, cursor, m["role"]))

    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    token_roles = [
        next((role for s, e, role in spans if s <= tok_s and tok_e <= e and tok_s < tok_e), None)
        for tok_s, tok_e in enc["offset_mapping"]
    ]
    return torch.tensor([enc["input_ids"]]), token_roles
