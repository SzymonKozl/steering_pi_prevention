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
from role_steering.steering import hf_upload_steered_outputs, steered_generate
from role_steering.eval.eval import eval_all
from role_steering.eval.llm_judge import judge_all


if modal.is_local():
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)


logger = logging.getLogger(__name__)


DEVICE = "cuda"


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
                [{"role": target_role, "text": text_sample}], tokenizer, max_tokens_per_message=cfg["probing"]["max_seqlen"]
            )
            content_idx = [i for i, r in enumerate(token_roles) if r == target_role]
            sample_acts = collect_layer_activations(model, input_ids.to(DEVICE), layers, token_indices=content_idx)
            for layer in layers:
                activations[layer].append(sample_acts[layer])
            labels += [target_role] * len(content_idx)
            groups += [f"{target_role}_{sample_ix}"] * len(content_idx)
    print(f"training classifiers for {model_name}")
    return train_role_probes({layer: torch.cat(acts) for layer, acts in activations.items()}, labels, groups)


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
        for alg_cfg in cfg["steering_algorithm"].values():
            for alg_params in _sweep_over_steering_alg(alg_cfg):
                for layer_set in cfg["layers"]:
                    new_cfg = deepcopy(cfg) | {"model": model, "steering_algorithm": alg_params, "layers": layer_set}
                    new_cfg.pop("models")
                    res.append(new_cfg)
                    if cfg["include_baseline"]:
                        res.append(deepcopy(new_cfg) | {"dont_steer": True})
    return res


def params_descr(alg_cfg: Dict[Any, str]) -> str:
    return "".join([alg_cfg["name"]] + [f"_{k}_{v}" for k, v in alg_cfg["params"].items()])


def cfg_to_descriptor(cfg: Dict[str, Any]) -> str:
    # converts sweep element to loggable key
    if not "dont_steer" in cfg:
        return f"{cfg['model']}_{params_descr(cfg['steering_algorithm'])}_{cfg['layers']}"
    else:
        return f"{cfg['model']}_baseline"


def run_single_model(cfg: Dict[str, Any], probes: Dict[int, Dict[str, torch.Tensor]], examples: List[SampleRunnable]) -> List[List[str]]:
    dont_steer = cfg.get('dont_steer', False)
    print(f"steering for {cfg['model']}; dont_steer={dont_steer}")
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
            operator=cfg["steering_algorithm"]["name"],
            alpha=cfg["steering_algorithm"]["params"]["alpha"],
            do_sample=True,
            temperature=example.temperature,
            max_new_tokens=cfg["max_new_tokens"],
            dont_steer=dont_steer
        )
        results.append(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:]))
    return results


def main(args: Namespace):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        # this flag may be added during later stages
        assert "dont_steer" not in cfg
    # 0. modal & wandb & hf init
    hf_login(token=os.environ["HF_TOKEN"])
    wandb.login(os.environ["WANDB_API_KEY"])
    wandb.init(
        entity=cfg["wandb_entity"],
        project=cfg["wandb_project"],
        name=cfg["run_name"]
    )
    wandb.log({"config": cfg})
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    wandb.log({"commit_sha": sha})
    img = modal_infra.create_image()
    app = modal_infra.get_app("role-probing-app", image=img)
    # 1. activation gathering
    gather_actiations_modal = modal_infra.func_wrap(gather_probes, app, cfg)
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
        sweep = generate_sweep(cfg["steering"])
        results = list(run_single_model_modal.starmap(
            [(c, extract_steering_vectors(probes[c["model"]])) for c in sweep],
            kwargs={"examples": cases}
        ))
        logger.info(f"uploading steered outputs to {cfg['steering']['outputs_repo']}")
        hf_upload_steered_outputs(cases, {cfg_to_descriptor(cfg_local): res for cfg_local, res in zip(sweep, results)}, cfg["steering"]["outputs_repo"])
        # 3. evaluating
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
                cfg_to_descriptor(cfg_local): eval_all(for_auto_eval, resp) for cfg_local, resp in zip(sweep, results)
            }
            wandb.log({"LLM auto results": wandb.Table(
                columns=["example_name", "desc"] + list(next(iter(results_auto.values())).keys()),
                data=[
                    [s.name, desc, *r.values()] for desc, ress in results_auto.items() for s, r in zip(for_auto_eval, ress)
                ]
            )})
    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help="path to the configuration yaml file")
    main(parser.parse_args())
