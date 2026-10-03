import torch
import torch.nn as nn
import pytest

from src.steering import (
    ActAdd,
    ProjectionSubtraction,
    SteeringHookManager,
    SteeringOperator,
    get_steering_operator,
    register_steering_operator,
)


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 16, returns_tuple: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.returns_tuple = returns_tuple

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        out = hidden_states * 1.0
        if self.returns_tuple:
            return (out, None)
        return out


class MockCausalLM(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_size: int = 16, returns_tuple: bool = False):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockDecoderLayer(hidden_size, returns_tuple=returns_tuple)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        batch, seq_len = input_ids.shape
        states = torch.zeros(batch, seq_len, 16)
        for layer in self.model.layers:
            states = layer(states)
            if isinstance(states, tuple):
                states = states[0]
        return states


def test_actadd_operator():
    op = ActAdd()
    states = torch.zeros(2, 4)
    vector = torch.ones(4)
    steered = op(states, vector, alpha=2.0)
    assert torch.allclose(steered, torch.full((2, 4), 2.0))


def test_projection_subtraction_operator():
    op = ProjectionSubtraction()
    states = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    vector = torch.tensor([1.0, 0.0])
    steered = op(states, vector, alpha=1.0)
    expected = torch.tensor([[0.0, 0.0], [0.0, 3.0]])
    assert torch.allclose(steered, expected)


def test_tokenwise_steering_multi_span():
    model = MockCausalLM(num_layers=2, hidden_size=16)
    token_roles = ["system", "system", "user", "user", "cot"]
    vec_system = torch.full((16,), 1.0)
    vec_user = torch.full((16,), 5.0)

    role_to_vector = {
        "system": vec_system,
        "user": vec_user,
        "cot": None,
    }

    dummy_input = torch.zeros(1, len(token_roles), dtype=torch.long)

    with SteeringHookManager(
        model=model,
        layers=[0],
        token_roles=token_roles,
        role_to_vector=role_to_vector,
        operator="ActAdd",
        alpha=1.0,
        use_cache=True,
    ):
        out = model(dummy_input)

    assert torch.allclose(out[0, 0:2], torch.full((2, 16), 1.0))
    assert torch.allclose(out[0, 2:4], torch.full((2, 16), 5.0))
    assert torch.allclose(out[0, 4], torch.zeros(16))


def test_layer_specific_vectors():
    model = MockCausalLM(num_layers=2, hidden_size=16)
    token_roles = ["user", "user"]

    role_to_vector = {
        0: {"user": torch.full((16,), 1.0)},
        1: {"user": torch.full((16,), 2.0)},
    }

    dummy_input = torch.zeros(1, 2, dtype=torch.long)

    with SteeringHookManager(
        model=model,
        layers=[0, 1],
        token_roles=token_roles,
        role_to_vector=role_to_vector,
        operator="ActAdd",
        alpha=1.0,
        use_cache=True,
    ):
        out = model(dummy_input)

    assert torch.allclose(out[0], torch.full((2, 16), 3.0))


def test_tuple_output_support():
    model = MockCausalLM(num_layers=2, hidden_size=16, returns_tuple=True)
    token_roles = ["user"]
    role_to_vector = {"user": torch.full((16,), 3.0)}
    dummy_input = torch.zeros(1, 1, dtype=torch.long)

    with SteeringHookManager(
        model=model,
        layers=[0],
        token_roles=token_roles,
        role_to_vector=role_to_vector,
        operator="ActAdd",
        alpha=1.0,
        use_cache=True,
    ):
        out = model(dummy_input)

    assert torch.allclose(out[0, 0], torch.full((16,), 3.0))


def test_caching_behavior():
    model = MockCausalLM(num_layers=1, hidden_size=16)
    token_roles = ["user", "user"]
    role_to_vector = {"user": torch.full((16,), 1.0)}

    mgr = SteeringHookManager(
        model=model,
        layers=[0],
        token_roles=token_roles,
        role_to_vector=role_to_vector,
        operator="ActAdd",
        alpha=1.0,
        use_cache=True,
    )

    with mgr:
        prefill_out = model(torch.zeros(1, 2, dtype=torch.long))
        assert torch.allclose(prefill_out[0], torch.full((2, 16), 1.0))

        decode_out = model(torch.zeros(1, 1, dtype=torch.long))
        assert torch.allclose(decode_out[0], torch.zeros(1, 16))


def test_no_cache_behavior():
    model = MockCausalLM(num_layers=1, hidden_size=16)
    token_roles = ["user", "user"]
    role_to_vector = {"user": torch.full((16,), 1.0)}

    mgr = SteeringHookManager(
        model=model,
        layers=[0],
        token_roles=token_roles,
        role_to_vector=role_to_vector,
        operator="ActAdd",
        alpha=1.0,
        use_cache=False,
    )

    with mgr:
        out1 = model(torch.zeros(1, 2, dtype=torch.long))
        assert torch.allclose(out1[0], torch.full((2, 16), 1.0))

        out2 = model(torch.zeros(1, 3, dtype=torch.long))
        assert torch.allclose(out2[0, 0:2], torch.full((2, 16), 1.0))
        assert torch.allclose(out2[0, 2], torch.zeros(16))


def test_custom_operator_registration():
    class DummyMultiplier(SteeringOperator):
        def __call__(self, states, vector, alpha):
            return states * alpha

    register_steering_operator("dummy_mult", DummyMultiplier())
    op = get_steering_operator("dummy_mult")
    assert isinstance(op, DummyMultiplier)


def test_steered_generate():
    from src.steering import steered_generate

    class GenerativeMockLM(MockCausalLM):
        def generate(self, input_ids, use_cache=True, **kwargs):
            # Calls forward internally to trigger registered hooks
            self.forward(input_ids)
            return torch.cat([input_ids, torch.tensor([[99]])], dim=1)

    model = GenerativeMockLM(num_layers=1, hidden_size=16)
    token_roles = ["user"]
    role_to_vector = {"user": torch.full((16,), 1.0)}

    out = steered_generate(
        model=model,
        input_ids=torch.zeros(1, 1, dtype=torch.long),
        layers=[0],
        token_roles=token_roles,
        role_to_vector=role_to_vector,
        operator="ActAdd",
        alpha=1.0,
        use_cache=True,
    )
    assert out.shape == (1, 2)
    assert out[0, 1].item() == 99
