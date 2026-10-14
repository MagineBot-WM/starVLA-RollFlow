"""Measure rolling prefetch at the target control Hz, with memory-only sensor/output.

Uses the actual model RPC and native EVA action path, but sends no ZMQ or
hardware commands and simulates no physics. Full ZMQ/GDK parity is a separate test.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from .runtime import DEFAULT_EVA_ROOT,load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eva-root',type=Path,default=DEFAULT_EVA_ROOT)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/agibot_g2_live.py')
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=10094)
    parser.add_argument('--actions',type=int,default=128)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.actions<16:parser.error('actions must be at least 16')
    config=load_config(args.config,args.eva_root)
    config.policy.host,config.policy.port=args.host,args.port
    replay=load_config(Path(__file__).parent/'configs/g2_replay.py',args.eva_root)
    replay.transport.dataset_dir=str(args.dataset.resolve());replay.transport.episode_id=0
    from core.registry import ROBOT_REGISTRY,TRANSPORT_REGISTRY,STRATEGY_REGISTRY
    from core.app.state import RuntimeState,SessionState,SessionMode
    from core.app.handlers.io import connect_policy,start_inference_loop
    from core.app.handlers.control import publish_next_action
    from transport.utils import _SimpleRate
    robot=ROBOT_REGISTRY.build(config.robot.type)
    spec=config.inference_strategies.rolling
    strategy=STRATEGY_REGISTRY.build(spec.type,**spec.args)
    source=TRANSPORT_REGISTRY.build(replay.transport.type,replay,robot)
    frames={}
    try:
        n_steps=len(source.get_action_trajectory())
        if args.actions>n_steps:parser.error('Requested actions exceed validation episode 0 length')
        for index in [0,*range(strategy.prefetch_after_steps,n_steps,16)]:
            source.seek(index);frames[index]=source.get_frame()
        prompt=str(source.current_task)
    finally:source.close()
    times=[]
    transport=SimpleNamespace(create_rate=lambda hz:_SimpleRate(hz),is_shutdown=lambda:False,
        is_offline=lambda:True,get_frame=lambda:frames[strategy._issued],
        set_action_issued_callback=lambda cb:setattr(transport,'callback',cb),
        reset_policy_chunk=lambda:None)
    def publish(action,target='sim'):
        times.append(time.monotonic());transport.callback()
    transport.publish_action=publish
    runtime=RuntimeState(robot=robot,transport=transport,infer_strategy=strategy)
    session=SessionState(mode=SessionMode.SIM,selected_task=prompt,is_setup_done=True)
    try:
        if not connect_policy(config,runtime,session):raise RuntimeError(runtime.last_policy_error)
        runtime.policy.reset(seed=7)
        start_inference_loop(config,runtime,session)
        rate=_SimpleRate(config.inference_cfg.publish_rate)
        while session.step_index<args.actions:
            if not publish_next_action(config,runtime,session):time.sleep(.001);continue
            rate.sleep()
        strategy.stop_loop()
        deltas=np.diff(times)*1000
        report=dict(physical_robot_actions=False,action_socket_created=False,physics_simulated=False,
            scope='native EVA scheduling + actual model RPC at target Hz; memory sensor/output excludes ZMQ/GDK overhead',
            config=str(args.config),target_hz=config.inference_cfg.publish_rate,commands_issued=len(times),
            session_steps=session.step_index,server=runtime.policy.metadata,requests=strategy.requests,gaps=strategy.gaps,
            action_interval_ms=dict(p50=float(np.percentile(deltas,50)),p95=float(np.percentile(deltas,95)),max=float(max(deltas))),
            chunk_boundary_intervals_ms=[float(deltas[i-1]) for i in range(16,len(times),16)])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['action_interval_ms']))
    finally:
        strategy.stop_loop()
        if runtime.policy:runtime.policy.close()


if __name__=='__main__':main()
