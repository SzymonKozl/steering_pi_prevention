from typing import List, Dict, Any
from argparse import ArgumentParser, Namespace
from copy import deepcopy
import sys
from pathlib import Path
import os
import logging

import yaml
import modal
from huggingface_hub import login as hf_login
import numpy as np
import torch
import wandb
import git
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(Path(__file__).parent.parent)
from src.infra import modal as modal_infra
from src.probing import hf_upload_probes, train_role_probes, collect_layer_activations, extract_steering_vectors
from src.data.dataset import dispatch_examples, SampleRunnable
from src.data.utils import fetch_samples_from_dataset, tokenize_with_role_preservation
from src.steering import hf_upload_steered_outputs, steered_generate
from src.eval.eval import eval_all
from src.eval.llm_judge import judge_all


logger = logging.getLogger(__name__)


DEVICE = "cuda"


def gather_probes(model_name: str, cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    probing_samples = fetch_samples_from_dataset(sample_no=cfg["probing"]["sample_no"], **cfg["probing"]["dataset"])
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    layers = cfg["probing"]["layers"]
    activations = {layer: [] for layer in layers}
    labels = []
    for target_role in cfg["roles"]:
        for text_sample in probing_samples:
            input_ids, token_roles = tokenize_with_role_preservation(
                [{"role": target_role, "text": text_sample}], tokenizer, max_tokens_per_message=cfg["probing"]["max_seqlen"]
            )
            content_idx = [i for i, r in enumerate(token_roles) if r == target_role]
            sample_acts = collect_layer_activations(model, input_ids.to(DEVICE), layers, token_indices=content_idx)
            for layer in layers:
                activations[layer].append(sample_acts[layer])
            labels += [target_role] * len(content_idx)
    return train_role_probes({layer: torch.cat(acts) for layer, acts in activations.items()}, labels)


def generate_sweep(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    res = []
    for model in cfg["models"]:
        new_cfg = deepcopy(cfg)
        new_cfg.pop("models")
        res.append(new_cfg | {"model": model})
    return res


def run_single_model(cfg: Dict[str, Any], probes: Dict[int, Dict[str, torch.Tensor]], examples: List[SampleRunnable]) -> List[List[str]]:
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], trust_remote_code=True).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    results = []
    for example in examples:
        input_ids, token_roles = tokenize_with_role_preservation(example.messages, tokenizer, add_generation_prompt=True)
        input_ids = input_ids.to(DEVICE).repeat(example.repeats, 1)
        outputs = steered_generate(
            model,
            input_ids,
            layers=list(probes.keys()),
            token_roles=[token_roles] * example.repeats,
            role_to_vector=probes,
            operator=cfg["steering_algorithm"],
            alpha=cfg["alpha"],
            do_sample=True,
            temperature=example.temperature,
            max_new_tokens=cfg["max_new_tokens"],
        )
        results.append(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:]))
    return results


def main(args: Namespace):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    # 0. modal & wandb & hf init
    hf_login(token=os.environ["HF_TOKEN"])
    wandb.init(
        entity=cfg["wandb_entity"],
        project=cfg["wandb_project"],
        name=cfg["run_name"]
    )
    wandb.log({"config": cfg})
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    wandb.log({"commit_sha": sha})
    app = modal_infra.get_app()
    img = modal_infra.create_image()
    # 1. activation gathering
    gather_actiations_modal = app.function(gather_probes, gpu=cfg["modal"]["gpu"], timeout=cfg["modal"]["timeout"], image=img)
    model_sweep = cfg["steering"]["models"]
    probes = list(gather_actiations_modal.map(model_sweep, kwargs={"cfg": cfg}))
    hf_upload_probes(cfg["probing"]["probes_repo"], {model: probe for model, probe in zip(model_sweep, probes)})
    # 2. steering
    cases = dispatch_examples(cfg["steering"])
    sweep = generate_sweep(cfg["steering"])
    run_single_model_modal = app.function(run_single_model, gpu=cfg["modal"]["gpu"], timeout=cfg["modal"]["timeout"], image=img)
    results = list(run_single_model_modal.starmap(
        [(cfg, extract_steering_vectors(prb)) for cfg, prb in zip(sweep, probes)],
        kwargs={"examples": cases}
    ))
    hf_upload_steered_outputs(cases, {model: res for model, res in zip(model_sweep, results)}, cfg["steering"]["outputs_repo"])
    # 3. evaluating
    for_lm_eval = [case for case in cases if case.desired_behaviour is not None]
    for_auto_eval = [case for case in cases if case.expected_output_value is not None]
    if for_lm_eval:
        judge_prompt = Path(cfg["eval"]["llm_as_a_judge_prompt"]).read_text()
        results_judge = {
            model: judge_all(for_lm_eval, [r for c, r in zip(cases, resp) if c.desired_behaviour is not None], judge_prompt, model=cfg["eval"]["llm_as_a_judge_model"])
            for model, resp in zip(model_sweep, results)
        }
        wandb.log({"LLM Judge results": wandb.Table(
            columns=["example_name", "model", "alignment_rate"],
            data=[
                [s.name, mdl, np.nanmean(r)] for mdl, ress in results_judge.items() for s, r in zip(for_lm_eval, ress)
            ]
        )})
    if for_auto_eval:
        results_auto = {
            model: eval_all(for_auto_eval, resp) for model, resp in zip(model_sweep, results)
        }
        wandb.log({"LLM auto results": wandb.Table(
            columns=["example_name", "model"] + list(next(iter(results_auto.values())).keys()),
            data=[
                [s.name, mdl, *r.values()] for mdl, ress in results_auto.items() for s, r in zip(for_auto_eval, ress)
            ]
        )})
    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help="path to the configuration yaml file")
    main(parser.parse_args())
