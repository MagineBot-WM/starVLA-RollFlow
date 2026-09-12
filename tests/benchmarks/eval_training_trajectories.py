"""Offline demonstration replay; action error is NOT closed-loop success.

Load one checkpoint, cache identical VLM features, and compare four decoders.
Source datasets are not modified. Only full H=32 windows are scored, C=8.
Rolling depth is distinct from per-request numerical integration steps.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import RollFlow


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1, help='First N training episodes per suite')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    wrapper = PolicyServerWrapper(args.checkpoint, use_bf16=True, unnorm_key='franka')
    framework = wrapper._framework
    head = framework.action_model
    cfg = head.rollflow.cfg
    assert (cfg.horizon, cfg.chunk_size, cfg.action_dim) == (32, 8, 7)
    processor = wrapper.get_norm_processor('franka')
    data_cfg = framework.config.datasets.vla_data
    report = {'checkpoint': args.checkpoint, 'seeds': args.seeds,
              'note': 'Offline training-set replay with teacher observations; not closed-loop success. Only full 32-frame windows.',
              'action_keys': processor.action_keys, 'results': []}
    original_predict = head.predict_action
    captured = {}

    # Capture exactly the production preprocessed VLM inputs to the action head.
    # This is local to this evaluation process, never edits the framework code.
    def capture(context, state=None, encoder_attention_mask=None, **kwargs):
        captured['context'] = context.detach().cpu()
        captured['mask'] = None if encoder_attention_mask is None else encoder_attention_mask.cpu()
        assert state is None, 'This LIBERO checkpoint is vision-only'
        return torch.zeros(1, 8, 7, device=context.device)

    variants = [(mode, n) for mode in ('euler', 'meanflow') for n in (1, 4, 8, 16)]
    variants += [(mode, k) for mode in ('rolling_instant', 'rolling_average') for k in (1, 2, 4)]
    for name, _, robot in DATASET_NAMED_MIXTURES[data_cfg.data_mix]:
        ds = make_LeRobotSingleDataset(Path(data_cfg.data_root_dir), name, robot, data_cfg=data_cfg)
        for episode, length in zip(ds.trajectory_ids[:args.episodes], ds.trajectory_lengths[:args.episodes]):
            anchors = list(range(0, int(length)-31, 8))
            if not anchors:
                continue
            features, targets = [], []
            head.predict_action = capture
            try:
                for index in anchors:
                    raw = ds.get_step_data(int(episode), index)
                    sample = ds._pack_sample(processor.transform(raw))
                    assert len(sample['image']) == 2
                    target = np.asarray(sample['action'], dtype=np.float32)[-32:]
                    assert target.shape == (32, 7) and np.isfinite(target).all()
                    if targets:
                        np.testing.assert_allclose(targets[-1][8:], target[:24], atol=1e-6)
                    targets.append(target)
                    framework.predict_action([sample])
                    features.append((captured['context'], captured['mask']))
            finally:
                head.predict_action = original_predict
            truth = np.concatenate([x[:8] for x in targets])
            folder = args.output / f'{name}_episode{episode}'
            folder.mkdir()
            arrays = {'target_normalized': truth, 'anchor_indices': np.array(anchors)}
            print(name, 'episode', episode, 'anchors', len(anchors), flush=True)
            for mode, steps in variants:
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    head.reset()
                    rolling = RollFlow(replace(cfg, inference_steps=steps)) if mode.startswith('rolling') else None
                    predictions = []
                    calls = 0

                    def velocity(z, s, t, context, **kwargs):
                        nonlocal calls
                        calls += 1
                        return head._predict_velocity(z, s, s if mode == 'rolling_instant' else t, context, **kwargs)

                    for context, mask in features:
                        context = context.to(head.device)
                        mask = None if mask is None else mask.to(head.device)
                        if rolling is not None:
                            pred = rolling.step(velocity, 1, context=context, device=head.device, dtype=head.dtype,
                                                state_features=None, encoder_attention_mask=mask)
                        else:
                            head.inference_decoder = mode
                            head.meanflow_steps = steps
                            pred = head.predict_action(context, encoder_attention_mask=mask)
                            calls += steps
                        predictions.append(pred[0].float().cpu().numpy())
                    pred = np.concatenate(predictions)
                    assert pred.shape == truth.shape and np.isfinite(pred).all()
                    error = pred-truth
                    key = f'{mode}_{steps}_seed{seed}'
                    arrays[key] = pred
                    # Native action coordinates: do not mix these units into a single MSE.
                    native = processor.unapply_actions(pred)
                    native_truth = processor.unapply_actions(truth)
                    entry = {'dataset': name, 'episode': int(episode), 'mode': mode, 'steps_or_depth': steps,
                             'seed': seed, 'frames': len(truth), 'head_calls': calls,
                             'normalized_mse': float(np.mean(error**2)),
                             'normalized_mse_per_dim': np.mean(error**2, axis=0).tolist(),
                             'native_rmse_per_dim': np.sqrt(np.mean((native-native_truth)**2, axis=0)).tolist(),
                             'gripper_sign_agreement': float(np.mean((pred[:,-1]>0)==(truth[:,-1]>0)))}
                    report['results'].append(entry)
                    print(key, entry['normalized_mse'], flush=True)
            np.savez_compressed(folder/'predictions.npz', **arrays)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(7, 1, figsize=(16, 16), sharex=True)
            for d, ax in enumerate(axes):
                ax.plot(truth[:,d], 'k', label='truth', linewidth=1.5)
                for mode in ('euler', 'meanflow', 'rolling_instant', 'rolling_average'):
                    ax.plot(arrays[f'{mode}_4_seed{args.seeds[0]}'][:,d], label=mode, alpha=.7, linewidth=.8)
                ax.set_ylabel(processor.action_keys[d])
            axes[0].legend()
            fig.suptitle(f'{name}, episode {episode}: normalized actions, 4 steps/depth')
            fig.tight_layout()
            fig.savefig(folder/'trajectory.png', dpi=130)
            plt.close(fig)
            (args.output/'report.json').write_text(json.dumps(report, indent=2))
        del ds
    print('DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
