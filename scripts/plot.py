"""
Standalone plotting: one grouped bar chart per evaluation metric of a finished run.
Bars are grouped by example and coloured by sweep item (hue = layer set, shade = steering params,
gray = baseline). Charts are saved locally and logged to a new wandb run named `<run_name>_bar_chart`
in the same group as the source run.
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import re
import tempfile
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import wandb
import yaml
from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# tables logged by scripts/run.py; every column other than these two is a metric
TABLE_KEYS = ["LLM Judge results", "LLM auto results"]
ID_COLUMNS = ["example_name", "desc"]

# categorical slots (fixed order), chart chrome
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE_COLOR = "#898781"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def find_run(api: wandb.Api, entity: str, project: str, run_name: str) -> "wandb.apis.public.Run":
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name}, order="-created_at")
    if len(runs) == 0:
        raise ValueError(f"no run named {run_name!r} in {entity}/{project}")
    if len(runs) > 1:
        print(f"{len(runs)} runs named {run_name!r}, using the latest ({runs[0].id})")
    return runs[0]


def fetch_tables(run: "wandb.apis.public.Run") -> Dict[str, Tuple[List[str], List[list]]]:
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for key in TABLE_KEYS:
            if key not in run.summary:
                print(f"run has no table {key!r}, skipping")
                continue
            path = run.summary[key]["path"]
            run.file(path).download(root=tmp, replace=True)
            table = json.loads((Path(tmp) / path).read_text())
            res[key] = (table["columns"], table["data"])
    return res


def split_descriptor(desc: str) -> Tuple[str, str]:
    """Sweep descriptor (see cfg_to_descriptor in run.py) -> (layer set, steering params); baseline -> ("baseline", "")."""
    if desc.endswith("_baseline"):
        return "baseline", ""
    m = re.fullmatch(r"(.*)_(\[[\d, ]*\])", desc)
    if m is None:
        return desc, ""
    prefix, layers = m.groups()
    # drop the model name: it ends at the first "_" after the org/ part
    params = prefix[prefix.find("_", prefix.rfind("/")) + 1:]
    return layers, params


def assign_colors(descs: List[str]) -> Dict[str, str]:
    """Hue per layer set (in sweep order), lighter -> darker shade per steering params, gray baseline."""
    parts = {d: split_descriptor(d) for d in descs}
    layer_sets = list(dict.fromkeys(l for l, _ in parts.values() if l != "baseline"))
    if len(layer_sets) > len(HUES):
        raise ValueError(f"{len(layer_sets)} layer sets but only {len(HUES)} distinguishable hues; plot fewer sweep items")
    colors = {}
    for layers, hue in zip(layer_sets, HUES):
        group = [d for d in descs if parts[d][0] == layers]
        for i, d in enumerate(group):
            # t in [-1, 1]: -1 -> tint toward the surface, 1 -> shade toward ink
            t = 0.0 if len(group) == 1 else 2 * i / (len(group) - 1) - 1
            base, target = np.array(to_rgb(hue)), np.array(to_rgb(SURFACE if t < 0 else INK))
            colors[d] = base + (target - base) * abs(t) * (0.45 if t < 0 else 0.35)
    for d in descs:
        if parts[d][0] == "baseline":
            colors[d] = BASELINE_COLOR
    return colors


def legend_label(desc: str) -> str:
    layers, params = split_descriptor(desc)
    return "baseline (no steering)" if layers == "baseline" else f"layers {layers}, {params}"


def plot_metric(title: str, examples: List[str], descs: List[str], values: Dict[Tuple[str, str], float],
                colors: Dict[str, str], out_path: Path):
    n_ex, n_desc = len(examples), len(descs)
    bar_w = 0.8 / n_desc
    fig, ax = plt.subplots(figsize=(max(8, 1.4 + n_ex * n_desc * 0.16), 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(n_ex)
    for j, d in enumerate(descs):
        xs = x - 0.4 + (j + 0.5) * bar_w
        ys = np.array([values.get((e, d), np.nan) for e in examples], dtype=float)
        ax.bar(xs, ys, width=bar_w, color=colors[d], edgecolor=SURFACE, linewidth=1, label=legend_label(d), zorder=2)
        # a zero score gets a stub so it is distinguishable from a missing one
        zero = ys == 0
        ax.hlines(np.zeros(zero.sum()), xs[zero] - bar_w * 0.4, xs[zero] + bar_w * 0.4, color=colors[d], linewidth=3, zorder=3)
        # missing score (e.g. the judge only answered "unclear")
        for xi in xs[np.isnan(ys)]:
            ax.text(xi, 0.02, "n/a", rotation=90, ha="center", va="bottom", fontsize=6, color=INK_SECONDARY)

    ax.set_xticks(x, [e.split("/")[-1] for e in examples], rotation=20, ha="right", color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(-0.5, n_ex - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(title.split(": ")[-1], color=INK_SECONDARY)
    ax.set_title(title, loc="left", color=INK, fontsize=13)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(color=colors[d], label=legend_label(d)) for d in descs],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(args: Namespace):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    entity, project = cfg["wandb_entity"], cfg["wandb_project"]
    wandb.login(key=os.environ["WANDB_API_KEY"])
    api = wandb.Api()
    src = find_run(api, entity, project, args.run_name)
    tables = fetch_tables(src)
    if not tables:
        raise ValueError(f"run {args.run_name!r} has no evaluation tables")

    # one color per sweep item, shared by all charts
    all_descs = list(dict.fromkeys(row[cols.index("desc")] for cols, data in tables.values() for row in data))
    # baseline first, then shades of one hue next to each other
    layer_order = list(dict.fromkeys(split_descriptor(d)[0] for d in all_descs))
    all_descs.sort(key=lambda d: (split_descriptor(d)[0] != "baseline", layer_order.index(split_descriptor(d)[0])))
    colors = assign_colors(all_descs)

    out_dir = Path(args.out_dir) / f"{args.run_name}_bar_chart"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = {}
    for key, (cols, data) in tables.items():
        ex_i, desc_i = cols.index("example_name"), cols.index("desc")
        examples = list(dict.fromkeys(row[ex_i] for row in data))
        descs = [d for d in all_descs if any(row[desc_i] == d for row in data)]
        for metric in [c for c in cols if c not in ID_COLUMNS]:
            m_i = cols.index(metric)
            values = {(row[ex_i], row[desc_i]): row[m_i] for row in data}
            title = f"{key}: {metric}"
            path = out_dir / f"{re.sub(r'[^\w]+', '_', title).strip('_').lower()}.png"
            examples += ["avg"]
            values |= {("avg", d): np.nanmean(np.array([values[k] for k in values.keys() if k[1] == d])) for d in descs}
            plot_metric(title, examples, descs, values, colors, path)
            charts[title] = path
            print(f"saved {path}")

    # same group as the source run; older runs were logged without one, so use their name as the group
    group = src.group or args.run_name
    wandb.init(entity=entity, project=project, name=f"{args.run_name}_bar_chart", group=group, job_type="plot",
               config={"source_run_id": src.id, "source_run_name": args.run_name})
    wandb.log({f"bar_chart/{title}": wandb.Image(str(path)) for title, path in charts.items()})
    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run-name', required=True, help="display name of the wandb run to plot")
    parser.add_argument('--config', default="config.yaml", help="config yaml to take wandb entity/project from")
    parser.add_argument('--out-dir', default="runs", help="local directory to save charts into")
    main(parser.parse_args())
