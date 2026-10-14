"""Small boundary adapter for upstream EVA APIs used by deployment.

The complete-snapshot reader is currently private in upstream EVA. Keep that
dependency here and fail explicitly if an upgrade changes it.
"""

import inspect
from functools import partial


def check_api() -> dict:
    from core.app.handlers.io import fetch_action_chunk, _loop_fetch_chunk, _loop_observation
    from core.app.handlers.control import publish_next_action
    from core.app.handlers.recording import build_policy_observation
    from core.config import load_config
    from transport.zmq import ZmqTransport, _ObservationReader
    from strategy.base_strategy import BaseInferStrategy
    required = {
        "fetch_action_chunk": (fetch_action_chunk, {"config", "runtime", "prompt", "prebuilt_observation"}),
        "_loop_fetch_chunk": (_loop_fetch_chunk, {"config", "runtime", "session", "prompt", "obs"}),
        "_loop_observation": (_loop_observation, {"config", "runtime", "prompt"}),
        "publish_next_action": (publish_next_action, {"config", "runtime", "session"}),
        "build_policy_observation": (build_policy_observation, {"frame", "prompt", "config", "runtime"}),
        "strategy._activate_chunk": (BaseInferStrategy._activate_chunk, {"action_chunk", "start_step"}),
        "strategy.take_or_fetch_chunk": (BaseInferStrategy.take_or_fetch_chunk, {"prompt", "fetch_chunk"}),
    }
    signatures = {}
    for name, (fn, arguments) in required.items():
        signature = inspect.signature(fn)
        if not arguments <= set(signature.parameters):
            raise RuntimeError(f"EVA API changed: {name}{signature}; update adapters/eva.py")
        signatures[name] = str(signature)
    if not callable(getattr(_ObservationReader, "_drain_latest", None)):
        raise RuntimeError("EVA snapshot reader changed; update adapters/eva.py before live deployment")
    if not all(callable(getattr(ZmqTransport, name, None)) for name in ("get_frame", "publish_action")):
        raise RuntimeError("EVA ZMQ transport API changed")
    if not callable(load_config):
        raise RuntimeError("EVA config loader API changed")
    return signatures


def latest_wire_observation(transport, reader_name="_reader"):
    reader = getattr(transport, reader_name, None)
    method = getattr(reader, "_drain_latest", None)
    if not callable(method):
        raise RuntimeError("EVA complete-snapshot API changed; update adapters/eva.py")
    return method()


def read_only_observation_reader(config, robot):
    import zmq
    from transport.zmq import _ObservationReader
    return _ObservationReader(config, robot, zmq, conflate=True)


def latest_only_robot_readers(transport):
    # Every policy message is a complete snapshot, so older queued images can be
    # dropped during inference. Recording retains its separate native reader.
    for name in ("_reader", "_qpos_reader"):
        getattr(transport, name).close()
        setattr(transport, name, read_only_observation_reader(transport._config, transport._robot))


def subscription_aware_publisher(transport):
    """XPUB preserves the native action bytes and exposes subscriber readiness."""
    import zmq
    transport._pub.close(linger=0)
    publisher = zmq.Context.instance().socket(zmq.XPUB)
    publisher.setsockopt(zmq.XPUB_VERBOSE, 1)
    publisher.setsockopt(zmq.LINGER, 0)
    if transport._config.transport.get("action_socket_role", "connect") == "bind":
        publisher.bind(transport._config.transport.pub_endpoint)
    else:
        publisher.connect(transport._config.transport.pub_endpoint)
    transport._pub = publisher
    return publisher


def runtime_for_fetch(fetch_chunk):
    """Resolve the runtime carried by EVA's native fetch callback."""
    if not isinstance(fetch_chunk, partial):
        return None
    bound = inspect.signature(fetch_chunk.func).bind_partial(*fetch_chunk.args, **fetch_chunk.keywords)
    runtime = bound.arguments.get("runtime")
    if runtime is None:
        raise RuntimeError("EVA fetch callback API changed; update adapters/eva.py")
    return runtime


def target_for_fetch(fetch_chunk):
    if not isinstance(fetch_chunk, partial):
        return "real"
    bound = inspect.signature(fetch_chunk.func).bind_partial(*fetch_chunk.args, **fetch_chunk.keywords)
    session = bound.arguments.get("session")
    mode = getattr(getattr(session, "mode", None), "value", None)
    return "sim" if mode == "sim" else "real"


def reset_callback_for_fetch(fetch_chunk):
    """Bind EVA's runtime so strategy reset also clears discarded policy chunks.

    Native setup invokes take_or_fetch_chunk for validation, discards that chunk,
    then resets the strategy. Direct evaluation callbacks reset policy explicitly.
    Keep partial/context inspection at this upstream boundary.
    """
    runtime = runtime_for_fetch(fetch_chunk)
    if runtime is None:
        return None
    def reset():
        reset_transport = getattr(runtime.transport, "reset_policy_chunk", None)
        if reset_transport is not None:
            reset_transport()
        policy = runtime.policy
        if policy is not None:
            policy.reset()
    return reset


def complete_dataset_frame(transport, camera_keys):
    """Decode recorded inputs without upstream's missing-video black-frame fallback."""
    from core.types import Observation
    required = ("_lock", "_frame_index", "_n_steps", "_obs_state", "_frame_source", "_frame_timestamp")
    if any(not hasattr(transport, name) for name in required):
        raise RuntimeError("EVA dataset reader API changed; update adapters/eva.py")
    with transport._lock:
        index = transport._frame_index
        if transport.is_shutdown() or index >= transport._n_steps:
            return None
        images = {}
        for key in camera_keys:
            source = transport._frame_source(key)
            image = source.read_at(index) if source is not None else None
            if image is None:
                raise ValueError(f"Dataset camera {key!r} cannot decode frame {index}; refusing substituted pixels")
            images[key] = image
        return Observation(images=images, state_qpos=transport._obs_state[index].copy(),
                           timestamp=transport._frame_timestamp(index))
