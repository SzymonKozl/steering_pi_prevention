import numpy as np
import torch
import torch.nn as nn
import pytest

from role_steering.probing import (
    ActivationCollector,
    collect_layer_activations,
    extract_role_vectors,
    extract_steering_vectors,
    load_steering_vectors,
    predict_role_probabilities,
    save_steering_vectors,
    train_linear_probe,
    train_role_probes,
    save_probes,
    load_probes,
)


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, *args, **kwargs):
        return self.linear(hidden_states)


class MockCausalLM(nn.Module):
    def __init__(self, num_layers=4, hidden_size=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockDecoderLayer(hidden_size) for _ in range(num_layers)]
        )
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, seq_len = input_ids.shape
        states = torch.randn(batch, seq_len, self.hidden_size)
        for layer in self.model.layers:
            states = layer(states)
        return states


def _make_separable_data(n_per_class=50, hidden_size=8):
    """Create linearly separable activation data for 3 roles."""
    rng = np.random.RandomState(42)
    acts_list, labels_list = [], []
    for i, role in enumerate(["system", "user", "assistant"]):
        center = np.zeros(hidden_size)
        center[i] = 5.0
        acts_list.append(rng.randn(n_per_class, hidden_size) + center)
        labels_list.extend([role] * n_per_class)
    acts = torch.tensor(np.vstack(acts_list), dtype=torch.float32)
    return acts, labels_list


def test_collect_layer_activations():
    model = MockCausalLM(num_layers=2, hidden_size=8)
    input_ids = torch.zeros(2, 5, dtype=torch.long)
    acts = collect_layer_activations(model, input_ids, layers=[0, 1])
    assert 0 in acts and 1 in acts
    assert acts[0].shape == (10, 8)  # 2 samples * 5 tokens
    assert acts[1].shape == (10, 8)


def test_collect_with_token_indices():
    model = MockCausalLM(num_layers=1, hidden_size=8)
    input_ids = torch.zeros(1, 10, dtype=torch.long)
    acts = collect_layer_activations(
        model, input_ids, layers=[0], token_indices=[0, 2, 4]
    )
    assert acts[0].shape == (3, 8)


def test_collect_with_per_sample_indices():
    model = MockCausalLM(num_layers=1, hidden_size=8)
    input_ids = torch.zeros(2, 6, dtype=torch.long)
    acts = collect_layer_activations(
        model, input_ids, layers=[0], token_indices=[[0, 1], [3, 4, 5]]
    )
    assert acts[0].shape == (5, 8)  # 2 + 3


def test_train_linear_probe():
    acts, labels = _make_separable_data()
    clf, acc = train_linear_probe(acts, labels, eval_fraction=0.2)
    assert acc > 0.8
    assert hasattr(clf, "predict")


def test_train_linear_probe_grouped_split():
    acts, labels = _make_separable_data()
    groups = [i // 10 for i in range(len(labels))]
    _, acc = train_linear_probe(acts, labels, groups=groups, eval_fraction=0.2)
    assert acc > 0.8


def test_train_role_probes():
    acts, labels = _make_separable_data()
    layer_acts = {0: acts, 5: acts}
    probes = train_role_probes(layer_acts, labels, eval_fraction=0.2)
    assert 0 in probes and 5 in probes
    assert probes[0]["accuracy"] > 0.8
    assert set(probes[0]["coef"]) == {"system", "user", "assistant"}
    assert set(probes[0]["intercept"]) == {"system", "user", "assistant"}


def test_extract_role_vectors_multiclass():
    acts, labels = _make_separable_data()
    classes, y = np.unique(labels, return_inverse=True)
    clf, _ = train_linear_probe(acts, y)
    vectors = extract_role_vectors(clf, classes.tolist())["coef"]
    assert "system" in vectors
    assert "user" in vectors
    assert "assistant" in vectors
    assert vectors["system"].shape == (8,)


def test_extract_role_vectors_binary():
    rng = np.random.RandomState(0)
    acts = torch.tensor(
        np.vstack([rng.randn(30, 4) + [3, 0, 0, 0], rng.randn(30, 4) - [3, 0, 0, 0]]),
        dtype=torch.float32,
    )
    labels = [1] * 30 + [0] * 30
    clf, _ = train_linear_probe(acts, labels, c_val=1.0)
    vectors = extract_role_vectors(clf, ["neg", "pos"])["coef"]
    assert "pos" in vectors and "neg" in vectors
    cos_sim = torch.dot(vectors["pos"], vectors["neg"]) / (
        vectors["pos"].norm() * vectors["neg"].norm()
    )
    assert cos_sim < -0.99  # should be opposite


def test_extract_steering_vectors():
    acts, labels = _make_separable_data()
    layer_acts = {0: acts, 7: acts}
    probes = train_role_probes(layer_acts, labels)
    sv = extract_steering_vectors(probes)
    assert 0 in sv and 7 in sv
    assert "user" in sv[0]


def test_save_load_steering_vectors(tmp_path):
    acts, labels = _make_separable_data()
    probes = train_role_probes({3: acts}, labels)
    sv = extract_steering_vectors(probes)

    path = tmp_path / "vecs.pt"
    save_steering_vectors(sv, path)
    loaded = load_steering_vectors(path)
    assert 3 in loaded
    assert torch.allclose(loaded[3]["user"], sv[3]["user"])


def test_save_load_probes(tmp_path):
    acts, labels = _make_separable_data()
    probes = train_role_probes({0: acts}, labels)

    path = tmp_path / "probes.joblib"
    save_probes(probes, path)
    loaded = load_probes(path)
    assert 0 in loaded
    assert loaded[0]["accuracy"] == probes[0]["accuracy"]


def test_predict_role_probabilities():
    acts, labels = _make_separable_data()
    probes = train_role_probes({0: acts}, labels)
    probs = predict_role_probabilities(probes[0], acts[:5])
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
