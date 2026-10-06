#!/usr/bin/env python3
"""Extract and validate standalone tactile encoders from official T-Rex weights."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'repos/T-Rex'))
from qwen_vla.DeformAE import DeformEncoder
from qwen_vla import Qwen3VLVLAModel
from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=PROJECT_ROOT / 'checkpoints/T-Rex/midtrain_source')
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'checkpoints/T-Rex/encoders')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model_path = args.source / 'model.pt'
    expected = '5c86368a2c8ee0e3d06638d5b64b8e30494e836b5776eecd8f93b6c94a84582b'
    actual = sha256(model_path)
    if actual != expected:
        raise ValueError('Source checkpoint SHA256 does not match pinned official revision')
    state = torch.load(model_path, map_location='cpu', weights_only=True, mmap=True)
    if 'state_dict' in state:
        state = state['state_dict']
    training = json.loads((args.source / 'training_args.json').read_text())
    vq_state = {key.removeprefix('tactile_vqvae.'): value.clone()
                for key, value in state.items() if key.startswith('tactile_vqvae.')}
    deform_state = {key.removeprefix('deform_encoder.'): value.clone()
                    for key, value in state.items() if key.startswith('deform_encoder.')}
    if not vq_state or not deform_state:
        raise ValueError('Missing tactile_vqvae or deform_encoder weights')
    stats = {f'tacf6_{name}': state[f'tacf6_vqvae_{name}'].clone()
             for name in ('min', 'max', 'mask')}
    if any(value.shape != (60,) for value in stats.values()):
        raise ValueError('Expected 60-channel F6 normalization statistics')
    vq_blob = {'config': training['vqvae_config'], 'model_state': vq_state, 'stats': stats}
    args.output.mkdir(parents=True, exist_ok=True)
    f6_path = args.output / 'f6_tactile_vqvae.pt'
    deform_path = args.output / 'sharpa_wave_deform_encoder.pth'
    torch.save(vq_blob, f6_path)
    torch.save(deform_state, deform_path)

    # Reload saved files and require complete state coverage, rather than relying
    # on the upstream permissive strict=False loaders.
    loaded_f6 = torch.load(f6_path, map_location='cpu', weights_only=True)
    loaded_deform = torch.load(deform_path, map_location='cpu', weights_only=True)
    for key, value in vq_state.items():
        assert torch.equal(value, loaded_f6['model_state'][key]), key
    for key, value in deform_state.items():
        assert torch.equal(value, loaded_deform[key]), key
    vq = TactileVQVAE(TactileVQVAEConfig.from_dict(loaded_f6['config'])).eval()
    vq.load_state_dict(loaded_f6['model_state'], strict=True)
    deform = DeformEncoder().eval()
    deform.load_state_dict(loaded_deform, strict=True)

    # Exercise the exact standalone loaders and raw-F6 normalization path used
    # by the official VLA with a tiny backbone; the encoders retain real weights.
    config = Qwen3VLTextConfig(vocab_size=128, hidden_size=256, intermediate_size=512,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=128)
    config.attention_bias = False
    model = Qwen3VLVLAModel(config, action_dim=62, use_tactile_deform=True,
                           use_tactile_vqvae=True, vqvae_config=loaded_f6['config']).eval()
    model.load_tactile_vqvae(str(f6_path))
    model.load_deform_encoder_weights(str(deform_path))
    torch.manual_seed(0)
    with torch.inference_mode():
        raw = torch.rand(1, 16, 10, 6)
        codes = model.encode_tactile_f6_history(raw)
        assert codes.shape == (1, 10)
        assert ((codes >= 0) & (codes < loaded_f6['config']['codebook_size'])).all()
        continuous = vq.encoder(torch.rand(1, 16, 5, 6))
        assert continuous.shape == (1, 5, loaded_f6['config']['embed_dim'])
        assert torch.isfinite(continuous).all()
        image = torch.rand(1, 1, 240, 240) * 255
        features = deform(image)
        assert features.shape == (1, 128, 15, 15)
        assert torch.isfinite(features).all()
        torch.testing.assert_close(model.deform_encoder(image), features, rtol=0, atol=0)
    manifest = {
        'created_at': datetime.datetime.now().astimezone().isoformat(),
        'source_repo': 'miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6',
        'source_revision': '62efb3bcb45a3df0e088c8909d759b582cfb98af',
        'source_model_sha256': actual,
        'source_model': str(model_path.relative_to(PROJECT_ROOT)),
        'extraction': 'Original tensor values and dtypes preserved; no retraining',
        'files': [{'name': path.name, 'size_bytes': path.stat().st_size,
                   'sha256': sha256(path)} for path in (f6_path, deform_path)],
        'vqvae_config': loaded_f6['config'],
        'tensor_counts': {'f6_vqvae': len(vq_state), 'deform_encoder': len(deform_state)},
        'validation': {'status': 'PASS', 'strict_state_loading': True,
                       'saved_tensors_equal_source': True, 'official_loaders': True,
                       'f6_raw_input_shape': list(raw.shape),
                       'f6_codes_shape': list(codes.shape),
                       'f6_continuous_features_shape': list(continuous.shape),
                       'deform_input_shape': list(image.shape),
                       'deform_features_shape': list(features.shape),
                       'device': 'cpu', 'inputs': 'synthetic; pretrained encoder weights'},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
