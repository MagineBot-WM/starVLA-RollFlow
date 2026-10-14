"""G2 recorded observations -> native hardware wire -> EVA -> model -> Mock GDK.

Observation/action sockets are restricted to temporary local IPC paths. No
physical SDK is loaded. This checks issued commands, not real robot dynamics.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from .runtime import DEFAULT_EVA_ROOT, load_config
from .evaluate_dataset_via_eva import DEFAULT_EPISODES, _metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eva-root", type=Path, default=DEFAULT_EVA_ROOT)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--episodes", default=",".join(map(str, DEFAULT_EPISODES)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args()
    package = Path(__file__).parent
    live = load_config(package / "configs/agibot_g2_live_sync.py", args.eva_root)
    replay = load_config(package / "configs/g2_replay.py", args.eva_root)
    live.policy.host, live.policy.port = args.host, args.port
    # The probe measures latency; a camera publisher is not backgrounded here.
    # Production retains the shorter freshness and inference limits in live.py.
    live.transport.max_frame_age_s = 30.0
    live.transport.trace_dir = str(args.output.parent / "wire_trace")
    live.policy.backend_options.trace_dir = str(args.output.parent / "policy_trace")
    live.policy.backend_options.trace_limit = 8
    live.policy.backend_options.max_inference_latency_s = None
    replay.transport.dataset_dir = str(args.dataset.resolve())
    from core.registry import ROBOT_REGISTRY, TRANSPORT_REGISTRY, STRATEGY_REGISTRY
    from core.app.state import RuntimeState, SessionState, SessionMode, SessionStatus
    from core.app.handlers.control import publish_next_action
    from core.app.handlers.io import connect_policy
    from .simulation.g2_wire import G2WireMock
    robot = ROBOT_REGISTRY.build(live.robot.type)
    results, arrays, latencies, intervals = [], {}, [], []
    total_chunks = 0
    episodes = [int(item) for item in args.episodes.split(",") if item.strip()]
    if not episodes or args.max_chunks < 0:
        parser.error("Pass episodes and a non-negative max-chunks")
    with tempfile.TemporaryDirectory(prefix="eva-wire-") as directory:
        node = G2WireMock(args.eva_root, directory)
        live.transport.sub_endpoint, live.transport.pub_endpoint = node.obs_endpoint, node.action_endpoint
        wire = TRANSPORT_REGISTRY.build(live.transport.type, live, robot)
        spec = live.inference_strategies.rolling
        strategy = STRATEGY_REGISTRY.build(spec.type, **spec.args)
        runtime = RuntimeState(robot=robot, transport=wire, infer_strategy=strategy)
        try:
            session = SessionState(mode=SessionMode.REAL)
            if not connect_policy(live, runtime, session):
                raise RuntimeError(runtime.last_policy_error)
            runtime.policy.reset(seed=7)
            for episode in episodes:
                replay.transport.episode_id = episode
                source = TRANSPORT_REGISTRY.build(replay.transport.type, replay, robot)
                try:
                    target = np.asarray(source.get_action_trajectory(), dtype=np.float32)
                    strategy.reset()
                    # First episode reset was already seeded above. Later reset
                    # clears caches without restarting the shared RNG sequence.
                    if strategy._reset_policy is None:
                        runtime.policy.reset()
                    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.RUNNING,
                        selected_task=str(source.current_task), is_setup_done=True)
                    predicted, truth = [], []
                    last_publish = None
                    start = time.perf_counter()
                    rate = wire.create_rate(live.inference_cfg.publish_rate)
                    for index in range(len(target)):
                        if session.chunk_index == 16 and args.max_chunks and total_chunks >= args.max_chunks:
                            break
                        # Policy observes only chunk boundaries. Reuse that
                        # recorded snapshot between requests while exercising
                        # every issued command; avoid 3 decoder seeks per action.
                        if session.action_chunk is None or session.chunk_index == 16:
                            source.seek(index)
                            frame = source.get_frame()
                        received = node.wait_frame(wire, frame)
                        np.testing.assert_array_equal(received.state_qpos, frame.state_qpos)
                        for key in frame.images:
                            np.testing.assert_array_equal(received.images[key], frame.images[key])
                        fetching = session.action_chunk is None or session.chunk_index == 16
                        if not publish_next_action(live, runtime, session):
                            raise RuntimeError(session.last_error or runtime.last_policy_error or "Native EVA did not issue an action")
                        command = node.consume_command()
                        np.testing.assert_array_equal(command, session.action_chunk[session.chunk_index - 1])
                        if fetching:
                            total_chunks += 1
                            latencies.append(runtime.policy.last_chunk["latency_ms"])
                        now = time.perf_counter()
                        if last_publish is not None:
                            intervals.append(dict(chunk_boundary=fetching, seconds=now-last_publish))
                        last_publish = now
                        predicted.append(command)
                        truth.append(target[index])
                        rate.sleep()
                    if not predicted:
                        raise RuntimeError("No actions checked")
                    arrays[f"episode_{episode}_prediction"] = np.asarray(predicted)
                    arrays[f"episode_{episode}_target"] = np.asarray(truth)
                    results.append(dict(episode=episode, elapsed_s=time.perf_counter()-start,
                                        **_metrics(np.asarray(predicted), np.asarray(truth))))
                    print(f"episode={episode} commands={len(predicted)} chunks_total={total_chunks}", flush=True)
                finally:
                    source.close()
                if args.max_chunks and total_chunks >= args.max_chunks:
                    break
            predictions = np.concatenate([arrays[k] for k in arrays if k.endswith("prediction")])
            targets = np.concatenate([arrays[k] for k in arrays if k.endswith("target")])
            report = dict(physical_robot_actions=False, endpoints="private local IPC", sdk="Mock GDK",
                observation_source="recorded dataset; not conditioned on simulated commands",
                observation_between_chunks="held recorded chunk-boundary snapshot with newly published wire timestamps",
                path="native G2 camera/state -> ZMQ -> EVA publish_next_action -> StarVLA -> EVA processing -> ZMQ -> native G2 servo request -> Mock GDK",
                server=runtime.policy.metadata, episodes=results, chunks=total_chunks,
                commands_received=len(node.received), servo_requests=len(node.servo_requests),
                command_max_abs_error=0.0, state_max_abs_error=0.0, pixel_max_abs_error=0,
                overall_metrics=_metrics(predictions, targets), inference_latency_ms=latencies,
                action_intervals=intervals, target_publish_hz=live.inference_cfg.publish_rate,
                guard_limits_overridden_for_synchronous_probe=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            if runtime.policy is not None:
                runtime.policy.close()
            wire.close()
            node.close()


if __name__ == "__main__":
    main()
