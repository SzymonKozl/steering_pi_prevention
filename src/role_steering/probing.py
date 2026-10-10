from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from huggingface_hub import create_repo, upload_file
import modal
if not modal.is_local():
    import tqdm

from role_steering.steering import get_model_layers


class ActivationCollector:
    def __init__(
        self,
        model: nn.Module,
        layers: List[int],
        token_indices: Optional[Union[List[int], List[List[int]]]] = None,
    ):
        """
        Args:
            model: Causal language model.
            layers: List of layer indices to collect activations from.
            token_indices: Token positions to collect per sample or globally.
        """
        self.model = model
        self.layers = layers
        self.token_indices = token_indices
        self.activations: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            states = output[0] if isinstance(output, tuple) else output
            batch_size, seq_len, _ = states.shape

            for b in range(batch_size):
                if self.token_indices is None:
                    self.activations[layer_idx].append(states[b].detach().cpu())
                else:
                    if (
                        self.token_indices
                        and isinstance(self.token_indices[0], list)
                    ):
                        indices = (
                            self.token_indices[b]
                            if b < len(self.token_indices)
                            else []
                        )
                    else:
                        indices = self.token_indices  # type: ignore

                    valid_idx = [i for i in indices if i < seq_len]
                    if valid_idx:
                        idx_tensor = torch.as_tensor(
                            valid_idx, device=states.device, dtype=torch.long
                        )
                        self.activations[layer_idx].append(
                            states[b, idx_tensor, :].detach().cpu()
                        )

        return hook

    def __enter__(self) -> "ActivationCollector":
        model_layers = get_model_layers(self.model)
        for layer_idx in self.layers:
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def get_stacked_activations(self) -> Dict[int, torch.Tensor]:
        """
        Returns:
            Dictionary mapping layer index to concatenated tensor of shape (N_tokens, hidden_dim).
        """
        result = {}
        for layer_idx, act_list in self.activations.items():
            if act_list:
                result[layer_idx] = torch.cat(act_list, dim=0)
            else:
                result[layer_idx] = torch.empty((0,))
        return result


def collect_layer_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    layers: List[int],
    attention_mask: Optional[torch.Tensor] = None,
    token_indices: Optional[Union[List[int], List[List[int]]]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Args:
        model: Causal language model.
        input_ids: Tokenized inputs tensor of shape (batch, seq_len).
        layers: Target layer indices.
        attention_mask: Optional attention mask.
        token_indices: Specific token indices to gather.
    """
    model.eval()
    with ActivationCollector(model, layers, token_indices) as collector:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        return collector.get_stacked_activations()


def _to_numpy(a: Any) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


def train_linear_probe(
    activations: torch.Tensor,
    labels: Union[List[Any], np.ndarray, torch.Tensor],
    groups: Optional[Union[List[Any], np.ndarray]] = None,
    eval_fraction: float = 0.1,
    random_state: int = 1234,
    c_val: float = 5e-3,
    max_iter: int = 2000,
) -> Tuple[Any, float]:
    """
    Args:
        activations: Activation tensor of shape (N, hidden_dim).
        labels: Integer label array of length N.
        groups: Optional sequence id per token; if given, the split is done on whole sequences.
        eval_fraction: Holdout fraction for evaluation.
        random_state: Seed for reproducible train/test split.
        c_val: Regularization parameter C for LogisticRegression.
        max_iter: Maximum optimization iterations.
    """
    x = activations.to(torch.float32).cpu().numpy()
    y = np.asarray(labels)

    if len(x) != len(y):
        raise ValueError(
            f"Shape mismatch: activations ({len(x)}) vs labels ({len(y)})"
        )

    if eval_fraction > 0.0 and groups is not None:
        groups = np.asarray(groups)
        _, test_groups = train_test_split(np.unique(groups), test_size=eval_fraction, random_state=random_state)
        is_test = np.isin(groups, test_groups)
        x_train, y_train, x_eval, y_eval = x[~is_test], y[~is_test], x[is_test], y[is_test]
    elif eval_fraction > 0.0:
        x_train, x_eval, y_train, y_eval = train_test_split(
            x,
            y,
            test_size=eval_fraction,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
    else:
        x_train, y_train = x, y
        x_eval, y_eval = x, y

    try:
        import cuml
        clf = cuml.linear_model.LogisticRegression(penalty="l2", C=c_val, max_iter=max_iter, fit_intercept=True)
    except ImportError as e:
        print(f"could not import cuml; falling back to sklearn! ({e})")
        clf = LogisticRegression(C=c_val, max_iter=max_iter)
    clf.fit(x_train, y_train)

    accuracy = float((_to_numpy(clf.predict(x_eval)) == y_eval).mean())
    return clf, accuracy


def train_role_probes(
    layer_activations: Dict[int, torch.Tensor],
    labels: Union[List[Any], np.ndarray],
    groups: Optional[Union[List[Any], np.ndarray]] = None,
    **probe_kwargs: Any,
) -> Dict[int, Dict[str, Any]]:
    """
    Args:
        layer_activations: Dictionary mapping layer indices to activation tensors.
        labels: Role label array corresponding to activations.
        groups: Optional sequence id per token, used for the train/test split.
        probe_kwargs: Passed to train_linear_probe.
    Returns:
        {layer: {"coef": {role: (D,) tensor}, "intercept": {role: float}, "accuracy": float}}
    """
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    results = {}
    for layer_idx, acts in tqdm.tqdm(layer_activations.items(), "training probes on layers"):
        clf, acc = train_linear_probe(acts, y, groups, **probe_kwargs)
        print(f"layer {layer_idx} probe accuracy: {acc:.3f}")
        results[layer_idx] = extract_role_vectors(clf, classes.tolist()) | {"accuracy": acc}
    return results


def extract_role_vectors(
    probe: Any,
    classes: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Args:
        probe: Fitted (cuml or sklearn) LogisticRegression classifier trained on integer labels.
        classes: Role name for each integer label.
    """
    coef = torch.as_tensor(_to_numpy(probe.coef_), dtype=torch.float32)
    intercept = _to_numpy(probe.intercept_).astype(float).reshape(-1)

    if len(classes) == 2:
        return {
            "coef": {str(classes[1]): coef[0], str(classes[0]): -coef[0]},
            "intercept": {str(classes[1]): intercept[0], str(classes[0]): -intercept[0]},
        }
    return {
        "coef": {str(c): coef[i] for i, c in enumerate(classes)},
        "intercept": {str(c): intercept[i] for i, c in enumerate(classes)},
    }


def extract_steering_vectors(
    role_probes: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Args:
        role_probes: Output dictionary from train_role_probes.
    """
    return {layer_idx: info["coef"] for layer_idx, info in role_probes.items()}


def save_steering_vectors(
    vectors: Dict[int, Dict[str, torch.Tensor]],
    path: Union[str, Path],
) -> Path:
    """
    Args:
        vectors: Role-to-vector mapping per layer.
        path: Destination .pt file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, dest)
    return dest


def load_steering_vectors(
    path: Union[str, Path],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Args:
        path: Path to saved .pt vectors file.
    """
    return torch.load(path, weights_only=True)


def save_probes(
    probes: Dict[int, Dict[str, Any]],
    path: Union[str, Path],
) -> Path:
    """
    Args:
        probes: Probes dictionary with fitted classifiers.
        path: Destination file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probes, dest)
    return dest


def load_probes(
    path: Union[str, Path],
) -> Dict[int, Dict[str, Any]]:
    """
    Args:
        path: Path to serialized probes file.
    """
    return joblib.load(path)


def predict_role_probabilities(
    probe: Dict[str, Any],
    activations: torch.Tensor,
) -> np.ndarray:
    """
    Args:
        probe: Single-layer entry from train_role_probes.
        activations: Activation tensor of shape (N, hidden_dim).
    Returns:
        (N, n_roles) probabilities, roles ordered as in probe["coef"].
    """
    w = torch.stack(list(probe["coef"].values()))
    b = torch.tensor(list(probe["intercept"].values()), dtype=torch.float32)
    return torch.softmax(activations.to(torch.float32) @ w.T + b, dim=-1).numpy()


def hf_upload_probes(repo_path: str, probes: Dict[str, Dict[int, Dict[str, Any]]]):
    repo_id = create_repo(
        repo_path,
        repo_type="model",
        exist_ok=True,
    ).repo_id

    for model_name, probes_for_model in probes.items():
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(probes_for_model, f.name)

            upload_file(
                path_or_fileobj=f.name,
                path_in_repo=f"probes_{model_name}.pt",
                repo_id=repo_id,
                repo_type="model",
            )
