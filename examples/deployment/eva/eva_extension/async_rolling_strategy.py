"""One-chunk lookahead; all sensor reads and actuator writes stay on the control thread."""

import copy
import dataclasses
import logging
import queue
import threading
import time
from typing import ClassVar

from core.registry import STRATEGY_REGISTRY
from policy_client.base import PolicyRequestError

from .rolling_strategy import StarVlaRollingStrategy
from examples.deployment.eva.bridge_contract import execution_spec
from examples.deployment.eva.adapters.eva import (
    reset_callback_for_fetch, runtime_for_fetch, target_for_fetch,
)


@STRATEGY_REGISTRY.register("StarVlaAsyncRollingStrategy")
@dataclasses.dataclass
class StarVlaAsyncRollingStrategy(StarVlaRollingStrategy):
    """Prefetch after N issued actions, activate only after all C actions issue.

    Observations are copied at the trigger (e.g. step 8); that request produces
    the next absolute action range (e.g. steps 16..31). This is lookahead with
    an older observation, not a prediction conditioned on a future observation.
    No prefix trim, retries, temporal averaging or cache speculation.
    """

    tune_fields: ClassVar[list[dict]] = []
    runs_background_loop: bool = dataclasses.field(default=True, init=False)
    _thread: threading.Thread | None = dataclasses.field(default=None, init=False, repr=False)
    _running: bool = dataclasses.field(default=False, init=False)

    def __post_init__(self):
        super().__post_init__()
        spec = execution_spec(self.contract)
        if spec.get("scheduling") != "async_prefetch":
            raise ValueError("Async rolling requires an async_prefetch contract")
        self.prefetch_after_steps = spec["prefetch_after_steps"]

    def start_loop(self, prompt, fetch_chunk, get_observation):
        if self._running:
            return
        self._control_thread = threading.get_ident()
        self._runtime = runtime_for_fetch(fetch_chunk)
        if self._runtime is None or not callable(getattr(self._runtime.transport, "set_action_issued_callback", None)):
            raise ValueError("Async rolling requires native EVA callbacks and an issued-action-aware transport")
        if self._runtime.policy is None or getattr(self._runtime.policy, "_closed", False):
            raise PolicyRequestError("Reconnect the policy before starting async rolling")
        if getattr(self._runtime.policy, "timeout_s", 5.0) > 5.0:
            raise ValueError("Async rolling requires policy timeout_s <= 5 for bounded stop/reset")
        self._policy = self._runtime.policy
        self._reset_policy = reset_callback_for_fetch(fetch_chunk)
        self._fetch = fetch_chunk
        self._observe = get_observation
        self._prompt = prompt
        self._target = target_for_fetch(fetch_chunk)
        self._jobs = queue.Queue(maxsize=1)
        self._results = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._error = None
        self._active = None
        self._awaiting_issue = False
        self._issued = 0
        self._in_chunk = 0
        self._gap_start = None
        self.requests = []
        self.gaps = []
        self._running = True
        self._runtime.transport.set_action_issued_callback(self.action_issued)
        try:
            self._enqueue(0)
            self._thread = threading.Thread(target=self._worker, name="eva-rolling-prefetch", daemon=True)
            self._thread.start()
        except BaseException:
            self.stop_loop()
            raise

    def _assert_control_thread(self):
        if threading.get_ident() != self._control_thread:
            raise RuntimeError("Sensor/control operations must stay on the EVA control thread")

    def _enqueue(self, action_start):
        self._assert_control_thread()
        observation = self._observe()
        if observation is None:
            raise ValueError("No complete observation at the prefetch trigger")
        record = dict(observation_step=self._issued, action_start=action_start,
                      observation_timestamp=observation.get("timestamp"),
                      source_timestamp=getattr(self._runtime.transport, "_last_source_timestamp", None),
                      prefetch_after_steps=self.prefetch_after_steps,
                      requested_at=time.monotonic())
        self.requests.append(record)
        self._trace("prefetch_requested", **record)
        logging.getLogger(__name__).info("ASYNC_PREFETCH observation_step=%d action_start=%d", self._issued, action_start)
        self._jobs.put_nowait((copy.deepcopy(observation), record))

    def _trace(self, event, **fields):
        trace = getattr(self._runtime.transport, "_execution_trace", None)
        if trace is not None:
            trace.write(event, **fields)

    def _worker(self):
        while not self._stop.is_set():
            try:
                observation, record = self._jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            if self._stop.is_set():
                return
            try:
                if self._runtime.policy is not self._policy:
                    raise PolicyRequestError("Policy connection changed during async inference")
                chunk = self._crop_chunk(self._fetch(self._prompt, observation))
                record["completed_at"] = time.monotonic()
                record["latency_ms"] = 1000 * (record["completed_at"] - record["requested_at"])
                identity = dict(getattr(self._policy, "last_chunk", {}) or {})
                identity.update(observation_step=record["observation_step"], action_start=record["action_start"])
                if not self._stop.is_set():
                    self._results.put_nowait((chunk, record["action_start"], identity))
            except BaseException as error:
                self._error = error
                return

    def pop_next_action(self):
        self._assert_control_thread()
        if self._error is not None:
            raise PolicyRequestError("Async rolling stopped; reset/reconnect before continuing") from self._error
        if self._runtime.policy is not self._policy or getattr(self._policy, "_closed", False):
            raise PolicyRequestError("Async policy connection changed/closed; reset/reconnect before continuing")
        if self._awaiting_issue:
            raise RuntimeError("Previous popped action was not issued; refusing to skip it")
        if self._active is None:
            try:
                chunk, start, identity = self._results.get_nowait()
            except queue.Empty:
                if self._gap_start is None:
                    self._gap_start = time.monotonic()
                    self._trace("async_wait_start", action_start=self._issued)
                return None
            if start != self._issued:
                raise RuntimeError("Prefetched action range does not match issued action count")
            if self._gap_start is not None:
                gap = dict(action_start=start, seconds=time.monotonic() - self._gap_start)
                self.gaps.append(gap)
                self._trace("async_wait_end", **gap)
                logging.getLogger(__name__).info("ASYNC_WAIT action_start=%d duration_ms=%.1f", start, gap["seconds"]*1000)
                self._gap_start = None
            begin = getattr(self._runtime.transport, "begin_policy_chunk", None)
            if begin is not None:
                begin(chunk, identity, target=self._target)
            self._active = chunk
            self._activate_chunk(chunk, start)
        self._awaiting_issue = True
        return self._active[self._in_chunk].copy()

    def action_issued(self):
        """After actual send/env.step. Errors latch; never undo a successful issue."""
        try:
            self._assert_control_thread()
            if not self._awaiting_issue:
                raise RuntimeError("Unexpected issued action outside the async rolling stream")
            self._awaiting_issue = False
            self._issued += 1
            self._in_chunk += 1
            if self._in_chunk == self.prefetch_after_steps:
                self._enqueue(self._current_chunk_start_step + self.execute_horizon)
            if self._in_chunk == self.execute_horizon:
                self._active = None
                self._in_chunk = 0
        except BaseException as error:
            self._error = error

    def is_loop_running(self):
        # Stay active even on worker failure so EVA cannot fall back to sync and
        # accidentally advance the model cache again. pop raises the error.
        return self._running

    def stop_loop(self):
        if hasattr(self, "_stop"):
            self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=6.0)
            if thread.is_alive():
                raise RuntimeError("Inference did not stop within 6s; do not reset a concurrent model stream")
        self._thread = None
        self._running = False
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.transport.set_action_issued_callback(None)
        self._fetch = self._observe = self._runtime = self._policy = None
        self._active = None
        self._awaiting_issue = False

    def reset(self):
        self.stop_loop()
        super().reset()
