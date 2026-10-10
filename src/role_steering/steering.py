import abc
import re
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import datasets

from role_steering.data.dataset import SampleRunnable

class SteeringOperator(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        states: torch.Tensor,
        vector: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Args:
            states: Hidden states tensor of shape (N, D).
            vector: Steering direction tensor of shape (D,).
            alpha: Steering strength multiplier.
        """
        pass


class ActAdd(SteeringOperator):
    def __call__(
        self,
        states: torch.Tensor,
        vector: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Args:
            states: Hidden states tensor of shape (N, D).
            vector: Steering direction tensor of shape (D,).
            alpha: Steering strength multiplier.
        """
        return states + alpha * vector.to(device=states.device, dtype=states.dtype)


class ProjectionSubtraction(SteeringOperator):
    def __call__(
        self,
        states: torch.Tensor,
        vector: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Args:
            states: Hidden states tensor of shape (N, D).
            vector: Steering direction tensor of shape (D,).
            alpha: Steering strength multiplier.
        """
        v = vector.to(device=states.device, dtype=states.dtype)
        norm_sq = torch.dot(v, v)
        if norm_sq == 0:
            return states
        projection = (torch.matmul(states, v) / norm_sq).unsqueeze(-1) * v
        return states - alpha * projection


_OPERATORS: Dict[str, SteeringOperator] = {
    "actadd": ActAdd(),
    "projectionsubtraction": ProjectionSubtraction(),
    "projection_subtraction": ProjectionSubtraction(),
}


def register_steering_operator(name: str, operator: SteeringOperator) -> None:
    """
    Args:
        name: Identifier name for the steering operator.
        operator: SteeringOperator instance.
    """
    _OPERATORS[name.lower()] = operator


def get_steering_operator(name_or_op: Union[str, SteeringOperator]) -> SteeringOperator:
    """
    Args:
        name_or_op: Operator name string or SteeringOperator instance.
    """
    if isinstance(name_or_op, SteeringOperator):
        return name_or_op
    key = name_or_op.lower()
    if key not in _OPERATORS:
        raise ValueError(
            f"Unknown steering operator '{name_or_op}'. Available: {list(_OPERATORS.keys())}"
        )
    return _OPERATORS[key]


def get_model_layers(model: nn.Module) -> nn.ModuleList:
    """
    Args:
        model: HuggingFace causal LM instance (e.g. GptOssForCausalLM, LlamaForCausalLM).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Could not locate decoder layers on {type(model).__name__}")


class SteeringHook:
    def __init__(
        self,
        role_to_vector: Dict[str, Optional[torch.Tensor]],
        token_roles: List[List[str]],
        operator: SteeringOperator,
        alpha: float = 1.0,
        use_cache: bool = True,
    ):
        """
        Args:
            role_to_vector: Mapping from role name to steering tensor or None.
            token_roles: Per-sample list of token roles.
            operator: SteeringOperator instance.
            alpha: Steering strength multiplier.
            use_cache: Whether generation uses KV caching.
        """
        self.role_to_vector = role_to_vector
        self.token_roles = token_roles
        self.operator = operator
        self.alpha = alpha
        self.use_cache = use_cache
        self.prefill_done = False

        self._role_indices: List[Dict[str, List[int]]] = []
        for sample_roles in token_roles:
            mapping: Dict[str, List[int]] = {}
            for idx, r in enumerate(sample_roles):
                mapping.setdefault(r, []).append(idx)
            self._role_indices.append(mapping)

    def reset(self) -> None:
        self.prefill_done = False

    def __call__(self, module: nn.Module, inputs, output):
        is_tuple = isinstance(output, tuple)
        states = output[0] if is_tuple else output
        batch_size, seq_len, _ = states.shape

        if self.use_cache and self.prefill_done:
            return output

        for b in range(batch_size):
            if b >= len(self._role_indices):
                continue

            sample_len = len(self.token_roles[b])
            if self.use_cache and seq_len < sample_len:
                continue

            for role, indices in self._role_indices[b].items():
                vector = self.role_to_vector.get(role)
                if vector is None:
                    continue

                valid_indices = [i for i in indices if i < seq_len]
                if not valid_indices:
                    continue

                idx_tensor = torch.as_tensor(
                    valid_indices, device=states.device, dtype=torch.long
                )
                states[b, idx_tensor, :] = self.operator(
                    states[b, idx_tensor, :], vector, self.alpha
                )

        if self.use_cache:
            self.prefill_done = True

        return (states, *output[1:]) if is_tuple else states


class SteeringHookManager:
    def __init__(
        self,
        model: nn.Module,
        layers: List[int],
        token_roles: Union[List[str], List[List[str]]],
        role_to_vector: Union[
            Dict[str, Optional[torch.Tensor]],
            Dict[int, Dict[str, Optional[torch.Tensor]]],
        ],
        operator: Union[str, SteeringOperator] = "ActAdd",
        alpha: float = 1.0,
        use_cache: bool = True,
    ):
        """
        Args:
            model: Causal language model.
            layers: Layer indices to hook.
            token_roles: Per-token role strings for single sequence or batch.
            role_to_vector: Role-to-vector mapping (per layer or uniform).
            operator: Steering algorithm name or operator.
            alpha: Steering strength multiplier.
            use_cache: Whether generation uses KV caching.
        """
        self.model = model
        self.layers = layers
        self.token_roles = (
            [token_roles] if token_roles and isinstance(token_roles[0], str) else token_roles  # type: ignore
        )
        self.role_to_vector = role_to_vector
        self.operator = get_steering_operator(operator)
        self.alpha = alpha
        self.use_cache = use_cache
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.hooks: List[SteeringHook] = []

    def _get_layer_vectors(self, layer_idx: int) -> Dict[str, Optional[torch.Tensor]]:
        if layer_idx in self.role_to_vector and isinstance(
            self.role_to_vector[layer_idx], dict
        ):
            return self.role_to_vector[layer_idx]  # type: ignore
        return self.role_to_vector  # type: ignore

    def register(self) -> "SteeringHookManager":
        model_layers = get_model_layers(self.model)
        for layer_idx in self.layers:
            layer = model_layers[layer_idx]
            hook = SteeringHook(
                role_to_vector=self._get_layer_vectors(layer_idx),
                token_roles=self.token_roles,
                operator=self.operator,
                alpha=self.alpha,
                use_cache=self.use_cache,
            )
            handle = layer.register_forward_hook(hook)
            self.hooks.append(hook)
            self.handles.append(handle)
        return self

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hooks.clear()

    def reset(self) -> None:
        for hook in self.hooks:
            hook.reset()

    def __enter__(self) -> "SteeringHookManager":
        return self.register()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()


def steered_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    layers: List[int],
    token_roles: Union[List[str], List[List[str]]],
    role_to_vector: Union[
        Dict[str, Optional[torch.Tensor]],
        Dict[int, Dict[str, Optional[torch.Tensor]]],
    ],
    attention_mask: Optional[torch.Tensor] = None,
    operator: Union[str, SteeringOperator] = "ActAdd",
    alpha: float = 1.0,
    use_cache: bool = True,
    dont_steer: bool = False,
    **generate_kwargs,
) -> torch.Tensor:
    """
    Args:
        model: Causal language model.
        input_ids: Tokenized input IDs tensor.
        layers: Layer indices to hook.
        token_roles: Per-token role strings.
        role_to_vector: Role-to-vector mapping.
        attention_mask: Optional attention mask tensor.
        operator: Steering algorithm name or operator.
        alpha: Steering strength multiplier.
        use_cache: Whether generation uses KV caching.
        generate_kwargs: Additional kwargs passed to model.generate.
    """
    if not dont_steer:
        with SteeringHookManager(
            model=model,
            layers=layers,
            token_roles=token_roles,
            role_to_vector=role_to_vector,
            operator=operator,
            alpha=alpha,
            use_cache=use_cache,
        ):
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                **generate_kwargs,
            )
    else:
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **generate_kwargs,
        )


def remove_special_chars_from_split_name(name: str) -> str:
    return re.sub(r'[^\w]', '_', name)


def hf_upload_steered_outputs(samples: List[SampleRunnable], outputs: Dict[str, List[List[str]]], repo_name: str):
    as_dict = {}
    for model, responses_by_model in outputs.items():
        rows = [
            sample.__dict__ | {"response": resp}
            for sample, responses in zip(samples, responses_by_model, strict=True) for resp in responses
        ]
        as_dict[remove_special_chars_from_split_name(model)] = datasets.Dataset.from_list(rows)
    ds = datasets.DatasetDict(as_dict)
    ds.push_to_hub(repo_name, private=False)
