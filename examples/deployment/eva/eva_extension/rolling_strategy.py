"""Synchronous execution for policies that advance an internal rolling cache."""

import dataclasses
import logging
from typing import Callable, ClassVar

import numpy as np

from core.registry import STRATEGY_REGISTRY
from strategy.base_strategy import BaseInferStrategy

from examples.deployment.eva.bridge_contract import execution_spec, load_contract


@STRATEGY_REGISTRY.register("StarVlaRollingStrategy")
@dataclasses.dataclass
class StarVlaRollingStrategy(BaseInferStrategy):
    # Cropping a rolling chunk would advance the model cache past unexecuted
    # actions. Keep this setting in the shared contract instead of the live UI.
    tune_fields: ClassVar[list[dict]] = []
    contract_path: str = ""
    _reset_policy: Callable[[], None] | None = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.contract = load_contract(self.contract_path)
        if execution_spec(self.contract)["kind"] != "rolling":
            raise ValueError("Use StarVlaChunkStrategy for a stateless policy")
        size = int(self.contract["action"]["chunk_size"])
        if self.execute_horizon is not None and self.execute_horizon != size:
            raise ValueError("Rolling execution horizon must equal the policy chunk size")
        self.execute_horizon = size
        if execution_spec(self.contract).get("timing") == "fixed_rate":
            # EVA's main loop already paces actions. A learned timed trajectory
            # must not add its native per-action arrival polling (up to 100 ticks).
            # Zero ticks skips that polling; it does not acknowledge arrival.
            self.sync_wait_max_ticks = 0
        super().__post_init__()

    def take_or_fetch_chunk(self, prompt, fetch_chunk):
        from examples.deployment.eva.adapters.eva import reset_callback_for_fetch
        callback = reset_callback_for_fetch(fetch_chunk)
        if callback is not None:
            self._reset_policy = callback
        chunk = super().take_or_fetch_chunk(prompt, fetch_chunk)
        from examples.deployment.eva.adapters.eva import runtime_for_fetch, target_for_fetch
        runtime = runtime_for_fetch(fetch_chunk)
        if runtime is not None and hasattr(runtime.transport, "begin_policy_chunk"):
            runtime.transport.begin_policy_chunk(chunk, getattr(runtime.policy, "last_chunk", None),
                                                 target=target_for_fetch(fetch_chunk))
        return chunk

    def reset(self):
        super().reset()
        if self._reset_policy is not None:
            self._reset_policy()

    def prepare_warmup_chunk(self, prompt, fetch_chunk):
        # EVA discards its prewarm predictions. Calling a stateful rolling policy
        # here would move its cache before any action was executed.
        logging.getLogger(__name__).info("Rolling policy: first inference occurs at execution start")
        return np.empty((0, int(self.contract["action"]["dim"])), dtype=np.float32)

    def _crop_chunk(self, action_chunk):
        expected = (int(self.contract["action"]["chunk_size"]),
                    int(self.contract["action"]["dim"]))
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if self.execute_horizon != expected[0] or chunk.shape != expected or not np.isfinite(chunk).all():
            raise ValueError(f"Rolling strategy requires a complete {expected} chunk")
        return super()._crop_chunk(chunk)


@STRATEGY_REGISTRY.register("StarVlaChunkStrategy")
@dataclasses.dataclass
class StarVlaChunkStrategy(BaseInferStrategy):
    """Stateless policies may execute a prefix of their full prediction horizon."""

    tune_fields: ClassVar[list[dict]] = []
    contract_path: str = ""

    def __post_init__(self):
        self.contract = load_contract(self.contract_path)
        spec = execution_spec(self.contract)
        if spec["kind"] != "chunk":
            raise ValueError("Rolling policies require StarVlaRollingStrategy")
        self.execute_horizon = spec["execute_steps"]
        super().__post_init__()

    def _crop_chunk(self, action_chunk):
        expected = (self.contract["action"]["chunk_size"], self.contract["action"]["dim"])
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.shape != expected or not np.isfinite(chunk).all():
            raise ValueError(f"Expected a finite policy chunk {expected}, got {chunk.shape}")
        return super()._crop_chunk(chunk)
