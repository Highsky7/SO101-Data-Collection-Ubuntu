"""Bounded, hardware-free CPU regression against the unchanged training inference.

Run on server2 only. Uses one existing task0 training frame, never cameras/motors.
Fresh child processes keep the training and inference dependency stacks separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

TRAIN = Path('/data/seungminlee/so101-rdvla-smolvla-h50-e20-20261108')
PYTHON = Path('/data/seungminlee/pi05-openarm-server2-after-rdvla-20261108/.venv/bin/python')
SEED = 1234


def evaluate(phase, directory, epoch):
    checkpoint = TRAIN / f"runs/smolvla/checkpoints/epoch-{epoch}"
    # No new runtime source/dependencies are visible in the reference child.
    if phase == 'reference':
        sys.path.insert(0, str(TRAIN / 'source'))
        import smolvla_adapter
    else:
        import so101_remote
        so101_remote.require_native_version()
    import numpy as np
    import torch

    torch.set_num_threads(4)
    if phase == 'reference':
        from so101_data import SO101Dataset, episode_arrays, inside, rgb_frame
        manifest = json.loads((TRAIN / 'data-manifest.json').read_text())
        episode = manifest['episodes'][0]
        assert episode['id'].startswith('so101_task0_') and episode['split'] == 'train'
        arrays = episode_arrays(str(inside(TRAIN / 'data', episode['data'])), episode['episode_index'])
        images = {
            name: rgb_frame(inside(TRAIN / 'data', item['path']), item['start'])
            for name, item in episode['videos'].items()
        }
        assert all(image.shape == (480, 640, 3) and image.dtype == np.uint8 for image in images.values())
        np.savez(directory / 'input.npz', state=arrays['observation.state'][0], actions=arrays['action'][:50],
                 task=episode['task'], episode=episode['id'], **images)
        adapter = smolvla_adapter.load(checkpoint / 'policy', 'cpu')
        # Reuse the actual training dataset/collation, including tensor layout,
        # rather than constructing a different noncontiguous image batch here.
        dataset = SO101Dataset(TRAIN / 'data', TRAIN / 'data-manifest.json')
        batch = torch.utils.data.default_collate([dataset[0]])
        prepared = adapter._prepare_inputs(batch)
        torch.manual_seed(SEED)
        rng = torch.get_rng_state()
        noise = adapter.model.model.sample_noise((1, 50, 32), 'cpu')
        torch.set_rng_state(rng)
        actions = adapter.predict(batch)
        # Exercise the original predict, then independently prove its sampled noise is fixed.
        with torch.no_grad():
            pixels, masks = adapter.model.prepare_images(prepared)
            explicit = adapter.model.model.sample_actions(
                pixels, masks, prepared['observation.language.tokens'],
                prepared['observation.language.attention_mask'], adapter.model.prepare_state(prepared), noise=noise,
            )[..., :6] * adapter._action_std + adapter._action_mean
        torch.testing.assert_close(actions, explicit, atol=1e-5, rtol=1e-5)
        np.savez(directory / 'reference.npz', actions=actions.cpu().numpy(), noise=noise.cpu().numpy(),
                 state=prepared['observation.state'].cpu().numpy(),
                 tokens=prepared['observation.language.tokens'].cpu().numpy())
    else:
        from lerobot.inference import ObservationSnapshot
        from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        with np.load(directory / 'input.npz', allow_pickle=False) as data, np.load(directory / 'reference.npz', allow_pickle=False) as reference:
            runner = so101_remote.make_runner(checkpoint, device='cpu', epoch=epoch)
            normalizer = next(step for step in runner.preprocessor.steps if isinstance(step, NormalizerProcessorStep))
            inverse = next(step for step in runner.postprocessor.steps if isinstance(step, UnnormalizerProcessorStep))
            assert {'observation.state', 'action'} <= normalizer._tensor_stats.keys()
            assert 'action' in inverse._tensor_stats
            observation = ObservationSnapshot({
                'observation.state': data['state'], 'observation.images.top': data['top'],
                'observation.images.wrist': data['wrist'],
            }, time.perf_counter(), str(data['task']), observation_id='training-frame-0')
            prepared = runner.preprocessor(runner._batch(observation))
            np.testing.assert_allclose(prepared['observation.state'].cpu().numpy(), reference['state'], atol=1e-6, rtol=0)
            np.testing.assert_array_equal(prepared['observation.language.tokens'].cpu().numpy(), reference['tokens'])
            torch.manual_seed(SEED)
            actions = runner.predict(observation).canonical_actions
            with torch.inference_mode():
                explicit = runner.postprocessor(runner.policy.predict_action_chunk(
                    prepared, noise=torch.from_numpy(reference['noise']),
                ))[0]
            torch.testing.assert_close(actions, explicit, atol=1e-5, rtol=1e-5)
            reference_actions = reference['actions'][0]
            actual = actions.cpu().numpy()
            recorded = data['actions']
            error = float(np.max(np.abs(actual - reference_actions)))
            result = {
                'checkpoint_epoch': epoch, 'checkpoint': str(checkpoint),
                'episode': str(data['episode']), 'frame_index': 0, 'seed': SEED,
                'device': 'cpu',
                'state_and_action_normalizer_stats_present': True,
                'normalized_state_and_tokens_match_training': True,
                'fixed_noise_matches_seeded_original_predict': True,
                'max_abs_prediction_difference': error,
                'equivalence_tolerance': 0.01,
                'numerical_equivalence': error <= 0.01,
                'joint_names': list(so101_remote.JOINTS),
                'units': 'first five joints: degrees; gripper: 0-100',
                'reference_mae_vs_recorded_per_joint': np.abs(reference_actions - recorded).mean(0).tolist(),
                'native_mae_vs_recorded_per_joint': np.abs(actual - recorded).mean(0).tolist(),
                'reference_max_abs_error_vs_recorded_per_joint': np.abs(reference_actions - recorded).max(0).tolist(),
                'native_max_abs_error_vs_recorded_per_joint': np.abs(actual - recorded).max(0).tolist(),
                'scope': 'one training frame only; not an independent evaluation or robot safety validation',
            }
            (directory / 'result.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
            assert error <= 0.01, 'Inference dependency/source migration changes the checkpoint predictions'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('reference', 'native'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epoch', type=int, choices=(10, 20), default=10)
    args = parser.parse_args()
    if args.phase:
        evaluate(args.phase, args.directory, args.epoch)
        return
    if args.output is None:
        parser.error('--output is required; existing results are never overwritten')
    with tempfile.TemporaryDirectory(prefix='so101-equivalence-') as directory:
        result_path = Path(directory) / 'result.json'
        with args.output.with_suffix('.log').open('x') as log:
            try:
                for phase in ('reference', 'native'):
                    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                   'CUDA_VISIBLE_DEVICES': ''}
                    environment['PYTHONPATH'] = str(TRAIN / 'runtime/smolvla-python-overlay') if phase == 'reference' else ''
                    subprocess.run([str(PYTHON), str(Path(__file__).resolve()), '--phase', phase,
                                    '--directory', directory, '--epoch', str(args.epoch)],
                                   check=True, env=environment, timeout=120, stdout=log, stderr=subprocess.STDOUT)
            finally:
                if result_path.exists():
                    with args.output.open('x') as result:
                        result.write(result_path.read_text())
        print(args.output.read_text(), flush=True)
        print('PASS: CPU training predict and native runner numerically equivalent; no hardware', flush=True)


if __name__ == '__main__':
    main()
