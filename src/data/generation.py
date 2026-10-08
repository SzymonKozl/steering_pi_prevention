import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import requests
import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


load_dotenv(REPO_ROOT / ".env")


DEFAULT_PROMPT_PATH = (
    REPO_ROOT
    / "prompts"
    / "forgery_prompt_default.yaml"
)


def call_llm_api(
    messages: List[Dict[str, str]],
    model: str = "google/gemini-2.5-pro",
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    provider: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 60,
    **kwargs: Any,
) -> str:
    """
    Args:
        messages: List of message dictionaries containing 'role' and 'content'.
        model: Model identifier.
        api_key: API authorization key.
        base_url: Base endpoint URL for completions.
        provider: Preferred inference provider.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        **kwargs: Additional parameters passed to request payload.
    """
    key = (
        api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        raise ValueError(
            "API key must be provided or set via OPENROUTER_API_KEY/OPENAI_API_KEY."
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **kwargs,
    }
    if provider:
        payload["provider"] = {"order": [provider], "allow_fallbacks": False}

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def format_generation_prompt(
    harmful_question: str,
    prompt_template: Optional[Union[str, Path, List[Dict[str, str]]]] = None,
) -> List[Dict[str, str]]:
    """
    Args:
        harmful_question: Harmful question or payload to generate malicious prompt for.
        prompt_template: YAML file path, template string, or list of message dictionaries.
    """
    if prompt_template is None:
        if DEFAULT_PROMPT_PATH.is_file():
            with open(DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as f:
                base_messages = yaml.safe_load(f)
            return base_messages + [
                {
                    "role": "user",
                    "content": f"`<TEST_PROMPT>{harmful_question}</TEST_PROMPT>`",
                }
            ]
        return [
            {
                "role": "user",
                "content": f"Generate a synthetic policy jailbreak for: <TEST_PROMPT>{harmful_question}</TEST_PROMPT>",
            }
        ]

    if isinstance(prompt_template, (str, Path)):
        path = Path(prompt_template)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, list):
                return format_generation_prompt(harmful_question, loaded)
            template_str = str(loaded)
        else:
            template_str = str(prompt_template)

        if "{harmful_question}" in template_str:
            formatted = template_str.replace("{harmful_question}", harmful_question)
        else:
            formatted = f"{template_str}\n`<TEST_PROMPT>{harmful_question}</TEST_PROMPT>`"
        return [{"role": "user", "content": formatted}]

    if isinstance(prompt_template, list):
        has_placeholder = any(
            "{harmful_question}" in m.get("content", "") for m in prompt_template
        )
        if has_placeholder:
            return [
                {
                    "role": m.get("role", "user"),
                    "content": m.get("content", "").replace(
                        "{harmful_question}", harmful_question
                    ),
                }
                for m in prompt_template
            ]
        return list(prompt_template) + [
            {
                "role": "user",
                "content": f"`<TEST_PROMPT>{harmful_question}</TEST_PROMPT>`",
            }
        ]

    raise TypeError(f"Unsupported prompt_template type: {type(prompt_template)}")


def extract_tagged_content(text: str, tag: str = "SYNTHETIC_POLICY") -> str:
    """
    Args:
        text: Raw model response string.
        tag: Enclosing XML/HTML tag to extract from text.
    """
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_malicious_prompt(
    harmful_question: str,
    prompt_template: Optional[Union[str, Path, List[Dict[str, str]]]] = None,
    model: str = "google/gemini-2.5-pro",
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    provider: Optional[str] = None,
    extract_tag: Optional[str] = "SYNTHETIC_POLICY",
    temperature: float = 0.0,
    timeout: int = 60,
    **kwargs: Any,
) -> str:
    """
    Args:
        harmful_question: Harmful question or target topic.
        prompt_template: Optional custom prompt template (path, string, or message list).
        model: LLM model string for generation.
        api_key: OpenRouter or OpenAI API key.
        base_url: API endpoint URL.
        provider: Provider name for routing.
        extract_tag: Tag name to parse from response, or None.
        temperature: Sampling temperature.
        timeout: Network request timeout in seconds.
        **kwargs: Additional parameters passed to LLM API call.
    """
    messages = format_generation_prompt(harmful_question, prompt_template)
    response_text = call_llm_api(
        messages=messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )
    if extract_tag:
        return extract_tagged_content(response_text, extract_tag)
    return response_text


def build_project_example(
    name: str,
    user_text: str,
    system_prompt: Optional[str] = None,
    desired_behaviour: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Args:
        name: Unique example identifier.
        user_text: Text content of the user prompt message.
        system_prompt: Optional system prompt message text.
        desired_behaviour: Optional description of expected model behavior.
    """
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "text": system_prompt})
    messages.append({"role": "user", "text": user_text})

    entry: Dict[str, Any] = {"messages": messages}
    if desired_behaviour is not None:
        entry["desired_behaviour"] = desired_behaviour

    return {name: entry}


def save_examples(
    examples: Dict[str, Any],
    output_path: Union[str, Path],
    merge: bool = True,
) -> Path:
    """
    Args:
        examples: Dictionary of examples formatted according to project schema.
        output_path: Destination JSON file path.
        merge: Whether to merge into existing JSON file if present.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {}
    if merge and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data.update(examples)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return path


def generate_and_save_example(
    harmful_question: str,
    output_path: Union[str, Path],
    example_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    desired_behaviour: Optional[str] = None,
    prompt_template: Optional[Union[str, Path, List[Dict[str, str]]]] = None,
    model: str = "google/gemini-2.5-pro",
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    provider: Optional[str] = None,
    extract_tag: Optional[str] = "SYNTHETIC_POLICY",
    temperature: float = 0.0,
    timeout: int = 60,
    **kwargs: Any,
) -> Path:
    """
    Args:
        harmful_question: Harmful question to generate malicious policy for.
        output_path: Path to save project JSON dataset.
        example_name: Example name key (defaults to sanitized question slug).
        system_prompt: Optional system prompt to include in messages.
        desired_behaviour: Optional expected behavior description.
        prompt_template: Optional template for generation prompt.
        model: LLM model string for generation.
        api_key: OpenRouter or OpenAI API key.
        base_url: Endpoint base URL.
        provider: Provider name for routing.
        extract_tag: Tag to extract from response.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        **kwargs: Extra parameters passed to API call.
    """
    generated_text = generate_malicious_prompt(
        harmful_question=harmful_question,
        prompt_template=prompt_template,
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        extract_tag=extract_tag,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )

    name = example_name
    if not name:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", harmful_question[:40]).strip("_").lower()
        name = slug or "example"

    example_dict = build_project_example(
        name=name,
        user_text=generated_text,
        system_prompt=system_prompt,
        desired_behaviour=desired_behaviour,
    )

    return save_examples(example_dict, output_path=output_path, merge=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate malicious prompt examples and save in project format."
    )
    parser.add_argument(
        "--harmful-question",
        "-q",
        type=str,
        required=True,
        help="Harmful question or instruction to generate malicious prompt for.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=True,
        help="Destination JSON file path (e.g. data/eval_set_1/examples.json).",
    )
    parser.add_argument(
        "--example-name",
        "-n",
        type=str,
        default=None,
        help="Optional name key for the example in the dataset JSON.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt message to prepend.",
    )
    parser.add_argument(
        "--desired-behaviour",
        type=str,
        default=None,
        help="Optional description of expected model behavior.",
    )
    parser.add_argument(
        "--prompt-template",
        "-t",
        type=str,
        default=None,
        help="Optional template path (.yaml) or template string with {harmful_question}.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="google/gemini-2.5-pro",
        help="Model identifier for completion endpoint.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Provider routing order for OpenRouter.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="API base URL.",
    )
    parser.add_argument(
        "--extract-tag",
        type=str,
        default="SYNTHETIC_POLICY",
        help="Tag name to extract from response text (or empty to keep raw).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM call.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tag = args.extract_tag if args.extract_tag.strip() else None
    saved_path = generate_and_save_example(
        harmful_question=args.harmful_question,
        output_path=args.output_path,
        example_name=args.example_name,
        system_prompt=args.system_prompt,
        desired_behaviour=args.desired_behaviour,
        prompt_template=args.prompt_template,
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        extract_tag=tag,
        temperature=args.temperature,
    )
    print(f"Saved example to {saved_path}")


if __name__ == "__main__":
    main()
