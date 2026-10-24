from typing import List, Dict, Any, Tuple
from argparse import ArgumentParser, Namespace
from copy import deepcopy
import sys
from pathlib import Path
import os
import logging
from itertools import product

import modal
import numpy as np
import torch
if modal.is_local():
    import wandb
    import git
    from huggingface_hub import login as hf_login
    from dotenv import load_dotenv
    import yaml
else:
    import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(Path(__file__).parent.parent)
from role_steering.infra import modal as modal_infra
from role_steering.probing import hf_upload_probes, hf_download_probes, train_role_probes, collect_layer_activations, extract_steering_vectors
from role_steering.data.dataset import dispatch_examples, SampleRunnable
from role_steering.data.utils import fetch_samples_from_dataset, tokenize_with_role_preservation
from role_steering.steering import hf_upload_steered_outputs, hf_download_steered_outputs, steered_generate, PROJECTION_SUBTRACTION_ROLE_MAP
from role_steering.eval.eval import eval_all, eval_agentdojo
from role_steering.ad_pipeline import (
    AgentDojoTask,
    SteeredLLM,
    build_agentdojo_pipeline,
    dispatch_agentdojo_tasks,
    hf_download_agentdojo_outputs,
    hf_upload_agentdojo_outputs,
    run_agentdojo_task,
)
from role_steering.eval.llm_judge import judge_all


if modal.is_local():
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)


logger = logging.getLogger(__name__)


DEVICE = "cuda"
# probing keeps all activations in RAM
GATHER_PROBES_KWARGS = {"memory": 65536}


def gather_probes(model_name: str, cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    print(f"collecting activations for {model_name}")
    probing_samples = fetch_samples_from_dataset(sample_no=cfg["probing"]["sample_no"], **cfg["probing"]["dataset"])
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
    print("loaded model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    layers = set.union(*[set(el) for el in cfg["steering"]["layers"]])
    activations = {layer: [] for layer in layers}
    labels = []
    groups = []
    for target_role in cfg["roles"]:
        print(f"role: {target_role}")
        for sample_ix, text_sample in enumerate(tqdm.tqdm(probing_samples, "forwarding calibration set...")):
            input_ids, token_roles = tokenize_with_role_preservation(
                [{"role": target_role, "text": text_sample}],
                tokenizer,
                max_tokens_per_message=cfg["probing"]["max_seqlen"],
                dummy_tool_calls=True,
            )
            content_idx = [i for i, r in enumerate(token_roles) if r == target_role]
            sample_acts = collect_layer_activations(model, input_ids.to(DEVICE), layers, token_indices=content_idx)
            for layer in layers:
                activations[layer].append(sample_acts[layer])
            labels += [target_role] * len(content_idx)
            groups += [f"{target_role}_{sample_ix}"] * len(content_idx)
    print(f"training classifiers for {model_name}")
    # concatenate one layer at a time so the per-sample lists are freed as we go
    for layer in layers:
        activations[layer] = torch.cat(activations[layer])
    return train_role_probes(activations, labels, groups)


def _sweep_over_steering_alg(alg_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    res = []
    keys = alg_cfg["params"].keys()
    for cfg in product(*[alg_cfg["params"][k] for k in keys]):
        new_cfg = deepcopy(alg_cfg)
        new_cfg["params"] = {k: v for (k, v) in zip(keys, cfg)}
        res.append(new_cfg)
    return res


def generate_sweep(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    res = []
    for model in cfg["models"]:
        model_cfgs = []
        for alg_cfg in cfg["steering_algorithm"].values():
            for alg_params in _sweep_over_steering_alg(alg_cfg):
                for layer_set in cfg["layers"]:
                    new_cfg = deepcopy(cfg) | {"model": model, "steering_algorithm": alg_params, "layers": layer_set}
                    new_cfg.pop("models")
                    model_cfgs.append(new_cfg)
        res += model_cfgs
        # baseline ignores steering params, so one per model is enough
        if cfg["include_baseline"]:
            res.append(deepcopy(model_cfgs[0]) | {"dont_steer": True})
    return res


def params_descr(alg_cfg: Dict[Any, str]) -> str:
    return "".join([alg_cfg["name"]] + [f"_{k}_{v}" for k, v in alg_cfg["params"].items()])


def cfg_to_descriptor(cfg: Dict[str, Any]) -> str:
    # converts sweep element to loggable key
    if not "dont_steer" in cfg:
        return f"{cfg['model']}_{params_descr(cfg['steering_algorithm'])}_{cfg['layers']}"
    else:
        return f"{cfg['model']}_baseline"


def dispatch_agentdojo(steering_cfg: Dict[str, Any]) -> List[AgentDojoTask]:
    ad_cfg = steering_cfg["samples"].get("agent_dojo")
    return dispatch_agentdojo_tasks(ad_cfg) if ad_cfg else []


def run_single_model(
    cfg: Dict[str, Any],
    probes: Dict[int, Dict[str, torch.Tensor]],
    examples: List[SampleRunnable],
    agentdojo_tasks: List[AgentDojoTask],
) -> Dict[str, List[List[Any]]]:
    """
    Returns {"direct_pi": responses per example, "agent_dojo": episode results per agentdojo task}.
    """
    dont_steer = cfg.get('dont_steer', False)
    print(f"steering for {cfg['model']}; dont_steer={dont_steer}")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], trust_remote_code=True).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    results = []
    for example in examples:
        input_ids, token_roles = tokenize_with_role_preservation(
            example.messages, tokenizer, add_generation_prompt=True, default_system_prompt=cfg["default_system_prompt"]
        )
        input_ids = input_ids.to(DEVICE).repeat(example.repeats, 1)
        outputs = steered_generate(
            model,
            input_ids,
            layers=cfg["layers"],
            token_roles=[token_roles] * example.repeats,
            role_to_vector=probes,
            operator=cfg["steering_algorithm"]["name"],
            alpha=cfg["steering_algorithm"]["params"]["alpha"],
            do_sample=True,
            temperature=example.temperature,
            max_new_tokens=cfg["max_new_tokens"],
            dont_steer=dont_steer
        )
        results.append(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:]))
    agentdojo_results = []
    if agentdojo_tasks:
        pipeline = build_agentdojo_pipeline(SteeredLLM(model, tokenizer, probes, cfg, temperature=cfg["temperature"]))
        for task in tqdm.tqdm(agentdojo_tasks, "agentdojo episodes"):
            agentdojo_results.append(run_agentdojo_task(pipeline, task))
    return {"direct_pi": results, "agent_dojo": agentdojo_results}


def dry_run(cfg: Dict[str, Any]):
    # prints everything that would be sent to modal, without touching modal / wandb / hf
    model_sweep = cfg["steering"]["models"]
    print("=== modal function settings ===")
    print(f"gather_probes: {modal_infra.function_kwargs(cfg, **GATHER_PROBES_KWARGS)}")
    print(f"run_single_model: {modal_infra.function_kwargs(cfg)}")
    print("\n=== 1. probing ===")
    if cfg["probing"]["try_reuse_repo"]:
        print(f"no modal call; would download probes for {model_sweep} from {cfg['probing']['probes_repo']}")
    else:
        layers = sorted(set.union(*[set(el) for el in cfg["steering"]["layers"]]))
        for model in model_sweep:
            print(f"gather_probes({model!r}): layers={layers}, roles={cfg['roles']}, "
                  f"sample_no={cfg['probing']['sample_no']}, max_seqlen={cfg['probing']['max_seqlen']}, dataset={cfg['probing']['dataset']}")
        print(f"would upload probes to {cfg['probing']['probes_repo']}")
    print("\n=== 2. steering: examples (passed to every job) ===")
    cases = dispatch_examples(cfg["steering"])
    for case in cases:
        print(f"{case.name}: repeats={case.repeats}, temperature={case.temperature}, "
              f"desired_behaviour={case.desired_behaviour is not None}, expected_output={case.expected_output_value!r}")
        for m in case.messages:
            print(f"    [{m['role']}] ({len(m['text'])} chars) {m['text'][:150]!r}")
    ad_tasks = dispatch_agentdojo(cfg["steering"])
    if ad_tasks:
        print(f"agent_dojo ({ad_tasks[0].benchmark_version}): repeats={ad_tasks[0].repeats}")
        for suite in sorted({t.suite for t in ad_tasks}):
            n_attacked = sum(t.suite == suite and t.injection_task is not None for t in ad_tasks)
            n_clean = sum(t.suite == suite and t.injection_task is None for t in ad_tasks)
            print(f"    {suite}: {n_attacked} attacked (user task, injection task) pairs, {n_clean} clean user tasks")
    print("\n=== 2. steering: jobs ===")
    sweep = generate_sweep(cfg["steering"])
    for i, c in enumerate(sweep):
        print(f"[{i}] {cfg_to_descriptor(c)}: model={c['model']}, layers={c['layers']}, "
              f"steering_algorithm={c['steering_algorithm']}, dont_steer={c.get('dont_steer', False)}, max_new_tokens={c['max_new_tokens']}, "
              f"default_system_prompt={c['default_system_prompt']}")
    n_gen = len(sweep) * sum(case.repeats for case in cases)
    n_episodes = len(sweep) * sum(t.repeats for t in ad_tasks)
    print(f"\ntotal: {len(sweep)} jobs x {len(cases)} examples -> {n_gen} generations of up to {cfg['steering']['max_new_tokens']} tokens")
    print(f"total: {len(sweep)} jobs x {len(ad_tasks)} agentdojo tasks -> {n_episodes} agent episodes")
    print(f"would upload outputs to {cfg['steering']['outputs_repo']}")


def evaluate(
    cfg: Dict[str, Any],
    cases: List[SampleRunnable],
    sweep: List[Dict[str, Any]],
    results: List[List[List[str]]],
    ad_tasks: List[AgentDojoTask],
    ad_results: List[List[List[Dict[str, Any]]]],
):
    for_lm_eval = [case for case in cases if case.desired_behaviour is not None]
    for_auto_eval = [case for case in cases if case.expected_output_value is not None]
    logger.info(f"LLM as a judge evaluation: ({len(for_lm_eval)} cases)")
    if for_lm_eval:
        judge_prompt = Path(cfg["eval"]["llm_as_a_judge_prompt"]).read_text()
        results_judge = {
            cfg_to_descriptor(cfg_local): judge_all(for_lm_eval, [r for c, r in zip(cases, resp) if c.desired_behaviour is not None], judge_prompt, model=cfg["eval"]["llm_as_a_judge_model"])
            for cfg_local, resp in zip(sweep, results)
        }
        wandb.log({"LLM Judge results": wandb.Table(
            columns=["example_name", "desc", "alignment_rate"],
            data=[
                [s.name, desc, np.nanmean(r)] for desc, ress in results_judge.items() for s, r in zip(for_lm_eval, ress)
            ]
        )})
    logger.info(f"Auto evaluation: ({len(for_auto_eval)} cases)")
    if for_auto_eval:
        results_auto = {
            cfg_to_descriptor(cfg_local): eval_all(for_auto_eval, [r for c, r in zip(cases, resp) if c.expected_output_value is not None], cfg)
            for cfg_local, resp in zip(sweep, results)
        }
        wandb.log({"LLM auto results": wandb.Table(
            columns=["example_name", "desc"] + list(next(iter(results_auto.values())).keys()),
            data=[
                [s.name, desc, *[np.mean(metric[i]) for metric in ress.values()]]
                for desc, ress in results_auto.items() for i, s in enumerate(for_auto_eval)
            ]
        )})
    logger.info(f"AgentDojo evaluation: ({len(ad_tasks)} tasks)")
    if ad_tasks:
        rows = [
            {"desc": cfg_to_descriptor(cfg_local)} | row
            for cfg_local, res in zip(sweep, ad_results) for row in eval_agentdojo(ad_tasks, res)
        ]
        wandb.log({"AgentDojo results": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})


def main(args: Namespace):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        # this flag may be added during later stages
        assert "dont_steer" not in cfg
    if args.dry_run:
        dry_run(cfg)
        return
    # 0. modal & wandb & hf init
    hf_login(token=os.environ["HF_TOKEN"])
    wandb.login(os.environ["WANDB_API_KEY"])
    wandb.init(
        entity=cfg["wandb_entity"],
        project=cfg["wandb_project"],
        name=cfg["run_name"],
        group=cfg["wandb_group"],
    )
    wandb.log({"config": cfg})
    wandb.log({"projection_subtraction_role_map": PROJECTION_SUBTRACTION_ROLE_MAP})
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    wandb.log({"commit_sha": sha})
    if cfg["eval"].get("eval_only"):
        # skip probing & steering, evaluate outputs already uploaded to outputs_repo
        cases = dispatch_examples(cfg["steering"])
        ad_tasks = dispatch_agentdojo(cfg["steering"])
        sweep = generate_sweep(cfg["steering"])
        descs = [cfg_to_descriptor(c) for c in sweep]
        print(f"eval only: loading outputs from {cfg['steering']['outputs_repo']}")
        outputs = hf_download_steered_outputs(cases, descs, cfg["steering"]["outputs_repo"]) if cases else {}
        ad_outputs = hf_download_agentdojo_outputs(ad_tasks, descs, cfg["steering"]["outputs_repo"]) if ad_tasks else {}
        sweep = [
            c for c in sweep
            if (not cases or cfg_to_descriptor(c) in outputs) and (not ad_tasks or cfg_to_descriptor(c) in ad_outputs)
        ]
        evaluate(
            cfg, cases, sweep,
            [outputs.get(cfg_to_descriptor(c), []) for c in sweep],
            ad_tasks,
            [ad_outputs.get(cfg_to_descriptor(c), []) for c in sweep],
        )
        wandb.finish()
        return
    img = modal_infra.create_image()
    app = modal_infra.get_app("role-probing-app", image=img)
    # 1. activation gathering
    gather_actiations_modal = modal_infra.func_wrap(gather_probes, app, cfg, **GATHER_PROBES_KWARGS)
    run_single_model_modal = modal_infra.func_wrap(run_single_model, app, cfg)
    model_sweep = cfg["steering"]["models"]
    with app.run():
        if cfg["probing"]["try_reuse_repo"]:
            print("trying to reuse probes from ", cfg["probing"]["probes_repo"])
            probes = hf_download_probes(cfg["probing"]["probes_repo"], model_sweep)
        else:
            probes_list = list(gather_actiations_modal.map(model_sweep, kwargs={"cfg": cfg}))
            probes = {mdl: prb for mdl, prb in zip(model_sweep, probes_list, strict=True)}
            hf_upload_probes(cfg["probing"]["probes_repo"], probes)
        # 2. steering
        cases = dispatch_examples(cfg["steering"])
        ad_tasks = dispatch_agentdojo(cfg["steering"])
        sweep = generate_sweep(cfg["steering"])
        results = list(run_single_model_modal.starmap(
            [(c, extract_steering_vectors(probes[c["model"]])) for c in sweep],
            kwargs={"examples": cases, "agentdojo_tasks": ad_tasks},
            return_exceptions=True
        ))
        # drop failed runs, keep the rest
        for c, r in zip(sweep, results):
            if isinstance(r, Exception):
                print(f"run {cfg_to_descriptor(c)} failed: {r!r}")
        sweep, results = zip(*[(c, r) for c, r in zip(sweep, results) if not isinstance(r, Exception)])
        logger.info(f"uploading steered outputs to {cfg['steering']['outputs_repo']}")
        if cases:
            hf_upload_steered_outputs(
                cases, {cfg_to_descriptor(cfg_local): res["direct_pi"] for cfg_local, res in zip(sweep, results)}, cfg["steering"]["outputs_repo"]
            )
        if ad_tasks:
            hf_upload_agentdojo_outputs(
                ad_tasks, {cfg_to_descriptor(cfg_local): res["agent_dojo"] for cfg_local, res in zip(sweep, results)}, cfg["steering"]["outputs_repo"]
            )
    # 3. evaluating
    evaluate(cfg, cases, sweep, [r["direct_pi"] for r in results], ad_tasks, [r["agent_dojo"] for r in results])
    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help="path to the configuration yaml file")
    parser.add_argument('--dry-run', action='store_true', help="print everything that would be sent to modal and exit")
    main(parser.parse_args())
