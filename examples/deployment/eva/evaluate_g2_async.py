"""Async G2 replay through native EVA/ZMQ/Mock GDK. No physical SDK or endpoints."""
import argparse
from functools import partial
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from .runtime import DEFAULT_EVA_ROOT, load_config
from .evaluate_dataset_via_eva import DEFAULT_EPISODES, _metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eva-root', type=Path, default=DEFAULT_EVA_ROOT)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/agibot_g2_live.py')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=10094)
    parser.add_argument('--episodes', default=','.join(map(str, DEFAULT_EPISODES)))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-actions', type=int, default=0)
    args = parser.parse_args()
    live = load_config(args.config, args.eva_root)
    live.policy.host, live.policy.port = args.host, args.port
    live.transport.max_frame_age_s = 30  # harness republishes on the control thread
    replay = load_config(Path(__file__).parent/'configs/g2_replay.py', args.eva_root)
    replay.transport.dataset_dir = str(args.dataset.resolve())
    from core.registry import ROBOT_REGISTRY, TRANSPORT_REGISTRY, STRATEGY_REGISTRY
    from core.app.state import RuntimeState, SessionState, SessionMode, SessionStatus
    from core.app.handlers.control import publish_next_action
    from core.app.handlers.io import connect_policy, _loop_fetch_chunk
    from core.app.handlers.recording import build_policy_observation
    from .simulation.g2_wire import G2WireMock
    robot = ROBOT_REGISTRY.build(live.robot.type)
    arrays, episodes, schedules, intervals = {}, [], [], []
    with tempfile.TemporaryDirectory(prefix='eva-async-wire-') as directory:
        node = G2WireMock(args.eva_root, directory)
        live.transport.sub_endpoint, live.transport.pub_endpoint = node.obs_endpoint, node.action_endpoint
        wire = TRANSPORT_REGISTRY.build(live.transport.type, live, robot)
        spec = live.inference_strategies.rolling
        strategy = STRATEGY_REGISTRY.build(spec.type, **spec.args)
        runtime = RuntimeState(robot=robot, transport=wire, infer_strategy=strategy)
        try:
            if not connect_policy(live, runtime, SessionState()):
                raise RuntimeError(runtime.last_policy_error)
            runtime.policy.reset(seed=7)
            for episode in map(int, args.episodes.split(',')):
                strategy.reset()
                replay.transport.episode_id = episode
                source = TRANSPORT_REGISTRY.build(replay.transport.type, replay, robot)
                try:
                    target = np.asarray(source.get_action_trajectory(), dtype=np.float32)
                    limit = min(len(target), args.max_actions or len(target))
                    # Decode video before the paced execution loop. Seeking a
                    # recorded stream at a trigger is not real sensor latency.
                    frames = {}
                    for index in [0, *range(strategy.prefetch_after_steps, len(target), 16)]:
                        source.seek(index)
                        frames[index] = source.get_frame()
                    session = SessionState(mode=SessionMode.REAL, status=SessionStatus.RUNNING,
                        selected_task=str(source.current_task), is_setup_done=True)
                    held = [None]
                    def observe():
                        index = strategy._issued
                        if index >= len(target):
                            return None
                        held[0] = frames[index]
                        frame = node.wait_frame(wire, held[0])
                        np.testing.assert_array_equal(frame.state_qpos, held[0].state_qpos)
                        for key in frame.images:
                            np.testing.assert_array_equal(frame.images[key], held[0].images[key])
                        return build_policy_observation(frame, session.selected_task, live, runtime)
                    strategy.start_loop(session.selected_task, partial(_loop_fetch_chunk, live, runtime, session), observe)
                    predicted, last = [], None
                    rate = wire.create_rate(live.inference_cfg.publish_rate)
                    while len(predicted) < limit:
                        node.wait_frame(wire, held[0])
                        if not publish_next_action(live, runtime, session):
                            time.sleep(0.005)
                            continue
                        command = node.consume_command()
                        np.testing.assert_array_equal(command, wire._pending_chunk[wire._issued_in_chunk-1])
                        predicted.append(command)
                        now = time.monotonic()
                        if last is not None:
                            intervals.append(dict(episode=episode, step=len(predicted)-1, seconds=now-last))
                        last = now
                        rate.sleep()
                    strategy.stop_loop()
                    if strategy._issued != len(predicted) or session.step_index != len(predicted):
                        raise RuntimeError('Native session/issued/received action counts differ')
                    arrays[f'episode_{episode}_prediction'] = np.asarray(predicted)
                    arrays[f'episode_{episode}_target'] = target[:limit]
                    episodes.append(dict(episode=episode, **_metrics(np.asarray(predicted), target[:limit])))
                    schedules.append(dict(episode=episode, requests=strategy.requests, gaps=strategy.gaps))
                    print(f'episode={episode} actions={len(predicted)} requests={len(strategy.requests)}', flush=True)
                finally:
                    strategy.stop_loop()
                    source.close()
            prediction = np.concatenate([v for k,v in arrays.items() if k.endswith('prediction')])
            truth = np.concatenate([v for k,v in arrays.items() if k.endswith('target')])
            report = dict(physical_robot_actions=False, sdk='Mock GDK', endpoints='private local IPC',
                config=str(args.config), server=runtime.policy.metadata, episodes=episodes, schedules=schedules,
                commands_received=len(node.received), servo_requests=len(node.servo_requests),
                command_max_abs_error=0.0, state_max_abs_error=0.0, pixel_max_abs_error=0,
                action_intervals=intervals, overall_metrics=_metrics(prediction, truth),
                observation_source='recorded state/images at issued-action count; no command-conditioned physics',
                observation_guard_override_s=30, production_inference_latency_guard_s=live.policy.backend_options.max_inference_latency_s,
                timing_scope='recorded trigger frames decoded before pacing; includes on-thread synthetic PNG publication and mock servo; not robot throughput')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
        finally:
            strategy.stop_loop()
            if runtime.policy is not None:
                runtime.policy.close()
            wire.close(); node.close()


if __name__ == '__main__':
    main()
