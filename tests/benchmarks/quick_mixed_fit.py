"""Bounded fixed-batch diagnostic; does not modify checkpoints or datasets."""
from pathlib import Path
import argparse
import numpy as np
import torch
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--g1-task', default='task_01_pick_the_orange')
    parser.add_argument('--g1-dataset', default='agibot-g1')
    parser.add_argument('--g1-robot-type', default='agibot-g1')
    parser.add_argument('--g1-only', action='store_true')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    torch.set_num_threads(2)
    root = Path('/data/tzq/starVLA_checkpoints')
    wrapper = PolicyServerWrapper(str(root / 'libero_agibot_g1_b32_c100_preflight/checkpoints/steps_2_pytorch_model.pt'), use_bf16=True, unnorm_key='franka')
    model = wrapper._framework
    head = model.action_model
    old = torch.load(root / 'libero_qwengroot_rollflow_h32_c8_b128_unfrozen_463a1f7/checkpoints/steps_55000_pytorch_model.pt', map_location='cpu', weights_only=True, mmap=True)
    head.load_state_dict({k.removeprefix('action_model.'): v for k, v in old.items() if k.startswith('action_model.')}, strict=False)
    del old
    batches = {}
    g1_name = f'{args.g1_dataset}/{args.g1_task}' if args.g1_task else args.g1_dataset
    for tag, name, robot in [('franka', 'libero/libero_spatial_no_noops_1.0.0_lerobot', 'libero_franka_rollflow'), ('agibot-g1', g1_name, args.g1_robot_type)]:
        if args.g1_only and tag != 'agibot-g1':
            continue
        ds = make_LeRobotSingleDataset(Path('/data/tzq/datasets/starVLA/Datasets'), name, robot, data_cfg={'include_state': True, 'video_backend': 'torchvision_av'})
        samples = [
            ds._pack_sample(ds.transforms(ds.get_step_data(int(ds.trajectory_ids[0]), i)))
            for i in (0, 8)
        ]
        actions = np.stack([s['action'] for s in samples])
        state = np.stack([s['state'] for s in samples]) if tag == 'agibot-g1' else None
        print('DATA', tag, actions.shape, 'finite', np.isfinite(actions).all(), 'range', actions.min(), actions.max(), 'views', [len(s['image']) for s in samples], 'state_range', None if state is None else (state.min(), state.max()), flush=True)
        inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=[s['image'] for s in samples], instructions=[s['lang'] for s in samples])
        with torch.no_grad():
            context = model.qwen_vl_interface(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1].detach().float()
        mask = inputs.get('attention_mask')
        batches[tag] = (context, torch.tensor(actions, device='cuda', dtype=torch.float32), None if state is None else torch.tensor(state, device='cuda', dtype=torch.float32), None if mask is None else mask.bool())
    head.float()
    initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    cases = [('agibot-g1',)] if args.g1_only else [('franka',), ('agibot-g1',), ('franka', 'agibot-g1')]
    for case in cases:
        head.load_state_dict(initial)
        head.train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        def loss(tag):
            context, actions, state, mask = batches[tag]
            torch.manual_seed(42)
            return head(context, actions, state, encoder_attention_mask=mask, embodiment=tag, training_step=1000)
        with torch.no_grad():
            before = {tag: float(loss(tag)) for tag in case}
        for step in range(args.steps):
            optimizer.zero_grad()
            total = sum(loss(tag) for tag in case) / len(case)
            total.backward()
            if step == 0:
                print('GRAD', case, {name: sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**0.5 for name, module in [('dit', head.model), ('libero', head.action_decoder), ('g1', head.adapters['agibot-g1'])]}, flush=True)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % 50 == 0:
                print('STEP', step + 1, 'loss', float(total.detach()), flush=True)
        with torch.no_grad():
            after = {tag: float(loss(tag)) for tag in case}
        print('FIT', case, before, after, flush=True)
        del optimizer


if __name__ == '__main__':
    main()
