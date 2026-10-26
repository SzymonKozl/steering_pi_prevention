import json
import random
import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import datasets
import torch
from agentdojo.agent_pipeline import AgentPipeline, BasePipelineElement, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt, _parse_model_output
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.types import ChatMessage, get_text_content_as_str
from torch import nn
from transformers import PreTrainedTokenizerBase

from role_steering.data.utils import tokenize_with_role_preservation
from role_steering.steering import get_model_layers, get_steering_operator, remove_special_chars_from_split_name, steered_generate

DEFAULT_BENCHMARK_VERSION = "v1.2.2"
# plain important_instructions addresses the model by name, which agentdojo only knows for API models
ATTACK = "important_instructions_no_model_name"
AGENTDOJO_CONFIG_NAME = "agent_dojo"
# roles that agentdojo conversations are tokenized into
PIPELINE_ROLES = ["system", "user", "assistant", "tool"]
STEERING_PARAMS = {"alpha"}


def _check_cfg(cfg: Dict[str, Any], model: nn.Module, vectors: Dict[int, Dict[str, torch.Tensor]]) -> None:
    for key in ["layers", "steering_algorithm", "max_new_tokens", "default_system_prompt"]:
        if key not in cfg:
            raise ValueError(f"missing config key '{key}'")
    layers = cfg["layers"]
    if not layers or not all(isinstance(l, int) for l in layers):
        raise ValueError(f"layers must be a non-empty list of ints, got {layers}")
    n_layers = len(get_model_layers(model))
    if any(not 0 <= l < n_layers for l in layers):
        raise ValueError(f"layers {layers} out of range for a model with {n_layers} layers")
    if cfg.get("dont_steer", False):
        return
    operator = get_steering_operator(cfg["steering_algorithm"]["name"])
    params = cfg["steering_algorithm"]["params"]
    if not set(params) <= STEERING_PARAMS:
        raise ValueError(f"unknown steering params {set(params) - STEERING_PARAMS}")
    # steering silently skips roles without a vector, so make sure every vector the operator asks for exists
    needed = {operator.vector_role(r) for r in PIPELINE_ROLES} - {None}
    for l in layers:
        if l not in vectors:
            raise ValueError(f"no steering vectors for layer {l}")
        missing = needed - set(vectors[l])
        if missing:
            raise ValueError(f"layer {l} is missing steering vectors for roles {missing}")


def _gptoss_parse_completion(completion: str) -> str:
    """
    Harmony output: an analysis message, then either a final answer or a tool call on the commentary channel. A tool
    call can use the prompted <function=...> format or gpt-oss's native recipient header
    ("<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>"); the latter is converted
    to the prompted format so _parse_model_output picks it up. Reasoning is dropped so it can't be parsed as a tool call.
    """
    last = completion.rsplit("<|start|>", 1)[-1]
    if "<|message|>" not in last:
        return ""
    header, body = last.split("<|message|>", 1)
    for stop in ["<|return|>", "<|end|>", "<|call|>"]:
        body = body.split(stop, 1)[0]
    recipient = re.search(r"to=functions\.([^\s<]+)", header)
    if recipient:
        return f"<function={recipient.group(1)}>{body.strip()}</function>"
    if "<|channel|>analysis" in header:
        # generation ran out of tokens before leaving the reasoning
        return ""
    return body


class SteeredLLM(BasePipelineElement):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        vectors: Dict[int, Dict[str, torch.Tensor]],
        cfg: Dict[str, Any],
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        """
        Args:
            model: Causal language model.
            tokenizer: Tokenizer of the model.
            vectors: Per-layer role-to-vector mapping.
            cfg: Single steering sweep element (see generate_sweep in scripts/run.py).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        _check_cfg(cfg, model, vectors)
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vectors = vectors
        self.cfg = cfg
        self.temperature = temperature
        self.top_p = top_p
        self.is_gptoss = "gpt-oss" in tokenizer.name_or_path

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # partially reused code from: https://github.com/ethz-spylab/agentdojo/blob/357d8ed603527858f40a7c42add7a866cd1f43c9/src/agentdojo/agent_pipeline/llms/local_llm.py
        messages_ = []
        for m in messages:
            role = m["role"]
            text = get_text_content_as_str(m["content"]) if m["content"] is not None else ""
            message = {"role": role, "text": text}
            if role == "system":
                message["text"] = _make_system_prompt(text, runtime.functions.values())
            if role == "tool":
                message["name"] = m["tool_call"].function
                if m.get("error") is not None:
                    message["text"] = json.dumps({"error": m["error"]})
                else:
                    message["text"] = json.dumps({"result": "Success" if text == "None" else text})
            messages_.append(message)

        input_ids, token_roles = tokenize_with_role_preservation(
            messages_,
            self.tokenizer,
            add_generation_prompt=True,
            default_system_prompt=self.cfg["default_system_prompt"],
        )
        input_ids = input_ids.to(self.model.device)

        generation = steered_generate(
            model=self.model,
            input_ids=input_ids,
            layers=self.cfg["layers"],
            token_roles=token_roles,
            role_to_vector=self.steering_vectors,
            operator=self.cfg["steering_algorithm"]["name"],
            dont_steer=self.cfg.get("dont_steer", False),
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.cfg["max_new_tokens"],
            **self.cfg["steering_algorithm"]["params"],
        )

        new_tokens = generation[0, input_ids.shape[1]:]
        if self.is_gptoss:
            completion = _gptoss_parse_completion(self.tokenizer.decode(new_tokens))
        else:
            completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        output = _parse_model_output(completion)
        return query, runtime, env, [*messages, output], extra_args


@dataclass
class AgentDojoTask:
    suite: str
    user_task: str
    # None = clean run without injections
    injection_task: Optional[str]
    repeats: int
    benchmark_version: str

    @property
    def name(self) -> str:
        return f"{self.suite}/{self.user_task}/{self.injection_task or 'clean'}"


def dispatch_agentdojo_tasks(config: Dict[str, Any]) -> List[AgentDojoTask]:
    """
    Samples fraction_used of the (user task, injection task) pairs in every suite, plus one clean run for every
    user task that got sampled.

    Args:
        config: The steering.samples.agent_dojo config section.
    """
    version = config.get("benchmark_version", DEFAULT_BENCHMARK_VERSION)
    suites = get_suites(version)
    if config["suites"] == "all":
        suite_names = sorted(suites)
    elif isinstance(config["suites"], list):
        suite_names = config["suites"]
    else:
        raise ValueError(f"suites must be a list or 'all', got {config['suites']!r}")
    unknown = set(suite_names) - set(suites)
    if unknown:
        raise ValueError(f"unknown agentdojo suites {unknown}, available in {version}: {sorted(suites)}")
    repeats = config.get("repeats", 1)
    attacked, clean = [], []
    for suite_name in suite_names:
        suite = suites[suite_name]
        pairs = sorted(product(suite.user_tasks, suite.injection_tasks))
        # seeding per suite keeps a suite's sample independent of which other suites are selected
        rng = random.Random(f"{config['seed']}/{suite_name}")
        no_to_sample = max(1, round(config["fraction_used"] * len(pairs))) if "fraction_used" in config else config["samples_per_suite"]
        sampled = sorted(rng.sample(pairs, no_to_sample))
        attacked += [AgentDojoTask(suite_name, u, i, repeats, version) for u, i in sampled]
        clean += [AgentDojoTask(suite_name, u, None, repeats, version) for u in sorted({u for u, _ in sampled})]
    return attacked + clean


class _RecordingPipeline(BasePipelineElement):
    # keeps the messages of the last query, which TaskSuite.run_task_with_pipeline doesn't return
    def __init__(self, pipeline: BasePipelineElement):
        self.pipeline = pipeline
        self.name = pipeline.name
        self.messages: Sequence[ChatMessage] = []

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        out = self.pipeline.query(query, runtime, env, messages, extra_args)
        self.messages = out[3]
        return out


def build_agentdojo_pipeline(llm: BasePipelineElement) -> _RecordingPipeline:
    tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_result_to_str), llm])
    pipeline = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm, tools_loop])
    # attacks look up the model name in the pipeline name; "local" is one agentdojo knows
    pipeline.name = "local"
    return _RecordingPipeline(pipeline)


def run_agentdojo_task(pipeline: _RecordingPipeline, task: AgentDojoTask) -> List[Dict[str, Any]]:
    """
    Returns one result per repeat: utility (user task solved), compliance (injection task carried out, None for
    clean runs), error (None unless the episode crashed) and the conversation as json.
    """
    suite = get_suite(task.benchmark_version, task.suite)
    user_task = suite.get_user_task_by_id(task.user_task)
    if task.injection_task is None:
        injection_task, injections = None, {}
    else:
        injection_task = suite.get_injection_task_by_id(task.injection_task)
        injections = load_attack(ATTACK, suite, pipeline).attack(user_task, injection_task)
    results = []
    for _ in range(task.repeats):
        pipeline.messages = []
        try:
            utility, security = suite.run_task_with_pipeline(pipeline, user_task, injection_task, injections)
            # agentdojo's security is True when the injection task succeeded
            result = {"utility": utility, "compliance": security if injection_task is not None else None, "error": None}
        except Exception as e:
            print(f"agentdojo episode {task.name} failed: {e!r}")
            result = {"utility": None, "compliance": None, "error": repr(e)}
        result["messages"] = json.dumps(pipeline.messages, default=lambda o: o.model_dump())
        results.append(result)
    return results


def hf_upload_agentdojo_outputs(tasks: List[AgentDojoTask], outputs: Dict[str, List[List[Dict[str, Any]]]], repo_name: str):
    # separate config so it lives next to the direct_pi outputs in the default config
    as_dict = {}
    for desc, results_by_desc in outputs.items():
        rows = [
            task.__dict__ | {"name": task.name} | result
            for task, results in zip(tasks, results_by_desc, strict=True) for result in results
        ]
        as_dict[remove_special_chars_from_split_name(desc)] = datasets.Dataset.from_list(rows)
    datasets.DatasetDict(as_dict).push_to_hub(repo_name, config_name=AGENTDOJO_CONFIG_NAME, private=False)


def hf_download_agentdojo_outputs(tasks: List[AgentDojoTask], descriptors: List[str], repo_name: str) -> Dict[str, List[List[Dict[str, Any]]]]:
    """Inverse of hf_upload_agentdojo_outputs; descriptors missing from the repo (e.g. failed runs) are skipped."""
    ds = datasets.load_dataset(repo_name, AGENTDOJO_CONFIG_NAME)
    res = {}
    for desc in descriptors:
        split = remove_special_chars_from_split_name(desc)
        if split not in ds:
            print(f"no agentdojo outputs for {desc} in {repo_name}, skipping")
            continue
        by_name = {}
        for row in ds[split]:
            by_name.setdefault(row["name"], []).append({k: row[k] for k in ["utility", "compliance", "error", "messages"]})
        res[desc] = [by_name[t.name] for t in tasks]
    return res
