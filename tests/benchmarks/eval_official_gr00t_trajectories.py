"""Official GR00T replay on the exact anchors of a prior RollFlow replay.

Uses the advertised primary-only image contract and default no-state loader
path. Compares native action coordinates, not different normalization spaces.
CPU VLM and GPU action head avoid displacing ongoing jobs. Not an LSD ablation.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    wrapper = PolicyServerWrapper(args.checkpoint, device='cpu', use_bf16=True, unnorm_key='franka',
        config_overrides=[f'framework.qwenvl.base_vlm={args.backbone}', 'framework.qwenvl.attn_implementation=sdpa'])
    model = wrapper._framework
    head = model.action_model
    head.to('cuda')
    proc = wrapper.get_norm_processor('franka')
    ref_report = json.loads((args.reference/'report.json').read_text())
    roll_proc = PolicyNormProcessor(ref_report['checkpoint'], unnorm_key='franka')
    data_cfg = model.config.datasets.vla_data
    data_cfg.data_root_dir = '/data/tzq/datasets/starVLA/Datasets/libero'
    data_cfg.load_all_data_for_training = False
    data_cfg.include_state = False
    data_cfg.action_mode = 'abs'
    data_cfg.video_backend = 'torchvision_av'
    report = {'checkpoint': args.checkpoint, 'camera': 'primary only, model-card contract',
              'state': 'not provided: packaged YAML does not enable include_state',
              'note': 'Native per-dimension errors on training observations; not success rate or controlled algorithm ablation.', 'results': []}
    captured = {}
    predict = head.predict_action

    def capture(context, state=None, encoder_attention_mask=None, **kwargs):
        captured['context'] = context
        captured['mask'] = encoder_attention_mask
        assert state is None
        # Framework converts this dummy return to NumPy; float32 is required.
        return torch.zeros(1, 8, 7)

    for source in sorted(args.reference.glob('*/predictions.npz')):
        name, ep = source.parent.name.rsplit('_episode', 1)
        ds = make_LeRobotSingleDataset(Path(data_cfg.data_root_dir), name, 'libero_franka', data_cfg=data_cfg)
        reference = np.load(source)
        features, targets = [], []
        head.predict_action = capture
        try:
            for anchor in reference['anchor_indices']:
                raw = ds.get_step_data(int(ep), int(anchor))
                sample = ds._pack_sample(proc.transform(raw))
                sample['image'] = sample['image'][:1]
                targets.append(np.asarray(sample['action'], dtype=np.float32))
                model.predict_action([sample])
                features.append((captured['context'], captured['mask']))
        finally:
            head.predict_action = predict
        truth = proc.unapply_actions(np.concatenate(targets))
        roll_truth = roll_proc.unapply_actions(reference['target_normalized'])
        np.testing.assert_allclose(truth, roll_truth, atol=2e-3, rtol=0)
        folder = args.output/source.parent.name
        folder.mkdir()
        arrays = {'truth_native': truth}
        for steps in (1, 4, 8, 16):
            head.num_inference_timesteps = steps
            for seed in ref_report['seeds']:
                torch.manual_seed(seed)
                out = [head.predict_action(c.to(head.device), encoder_attention_mask=None if mask is None else mask.to(head.device))[0].float().cpu().numpy() for c, mask in features]
                pred = proc.unapply_actions(np.concatenate(out))
                assert np.isfinite(pred).all()
                key = f'gr00t_{steps}_seed{seed}'
                arrays[key] = pred
                report['results'].append({'dataset': name, 'episode': int(ep), 'steps': steps, 'seed': seed,
                    'frames': len(truth), 'native_rmse_per_dim': np.sqrt(np.mean((pred-truth)**2, axis=0)).tolist()})
                print(name, key, report['results'][-1]['native_rmse_per_dim'], flush=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(7, 1, figsize=(16, 16), sharex=True)
        seed = ref_report['seeds'][0]
        for d, ax in enumerate(axes):
            ax.plot(truth[:,d], 'k', label='truth')
            ax.plot(arrays[f'gr00t_4_seed{seed}'][:,d], label='Official GR00T 30k, 4 steps')
            for mode in ('euler', 'meanflow', 'rolling_instant', 'rolling_average'):
                pred = roll_proc.unapply_actions(reference[f'{mode}_4_seed{seed}'])
                ax.plot(pred[:,d], label=f'RollFlow 50k {mode}', alpha=.65, linewidth=.8)
            ax.set_ylabel(proc.action_keys[d])
        axes[0].legend(fontsize=8)
        fig.suptitle('Native actions; different backbones/input contracts — not an LSD ablation')
        fig.tight_layout()
        fig.savefig(folder/'comparison.png', dpi=130)
        plt.close(fig)
        np.savez_compressed(folder/'predictions.npz', **arrays)
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        del ds
    print('DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
