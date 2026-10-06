from __future__ import annotations
from collections import deque
import numpy as np
import torch
from torch import nn
from .common import ROOT, sha
from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig
from qwen_vla.DeformAE import DeformEncoder


class FrozenEncoders(nn.Module):
    """Right-hand F6 continuous pre-quantization features and pooled T-Rex deform features."""
    def __init__(self, f6_path=None, deform_path=None):
        super().__init__()
        # Keep frozen feature extraction stable across batch sizes and CPU/GPU.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        self.f6_path = f6_path or ROOT / 'checkpoints/T-Rex/encoders/f6_tactile_vqvae.pt'
        self.deform_path = deform_path or ROOT / 'checkpoints/T-Rex/encoders/sharpa_wave_deform_encoder.pth'
        blob = torch.load(self.f6_path, map_location='cpu', weights_only=True)
        self.vq = TactileVQVAE(TactileVQVAEConfig.from_dict(blob['config']))
        self.vq.load_state_dict(blob['model_state'], strict=True)
        self.deform = DeformEncoder()
        self.deform.load_state_dict(torch.load(self.deform_path, map_location="cpu", weights_only=True), strict=True)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        for key in ('min', 'max', 'mask'):
            self.register_buffer(key + '_f6', torch.as_tensor(blob['stats']['tacf6_' + key])[30:60].reshape(5, 6))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Batch/state statistics of frozen modules must never change.
        return super().train(False)

    def normalize(self, raw):
        value = (2 * (raw.float() - self.min_f6) / (self.max_f6 - self.min_f6 + 1e-8) - 1).clamp(-1, 1)
        return torch.where(self.mask_f6, value, raw.float())

    @torch.no_grad()
    def f6_features(self, raw):
        if raw.ndim != 4 or tuple(raw.shape[1:]) != (16, 5, 6):
            raise ValueError('F6 input must be [B,16,5,6]')
        # The released VQ-VAE is per-finger, so 5 x 256 = 1280D.
        return self.vq.encoder(self.normalize(raw)).flatten(1)

    @torch.no_grad()
    def deform_features(self, images):
        if images.ndim != 5 or tuple(images.shape[1:]) != (5, 1, 240, 240):
            raise ValueError('Deform input must be [B,5,1,240,240]')
        features = self.deform(images.float().flatten(0, 1))
        return self.pool(features).reshape(images.shape[0], 2560)


class FrameProbe(nn.Module):
    def __init__(self, input_kind, head_kind, hidden=128, layers=1, num_classes=2):
        super().__init__()
        if input_kind not in ('f6', 'deform', 'f6_deform') or head_kind not in ('mlp', 'lstm'):
            raise ValueError('Unknown experimental group')
        self.input_kind, self.head_kind = input_kind, head_kind
        self.hidden, self.layers = hidden, layers
        self.f6_projection = nn.Linear(1280, 128) if input_kind != 'deform' else None
        self.deform_projection = nn.Linear(2560, 128) if input_kind != 'f6' else None
        width = 256 if input_kind == 'f6_deform' else 128
        self.context = nn.LSTM(width, hidden, layers, batch_first=True, bidirectional=False,
                               dropout=0.1 if layers > 1 else 0) if head_kind == 'lstm' else nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.num_classes = num_classes
        self.prediction = nn.Linear(hidden, num_classes)
        for key, dim in [('f6', 1280), ('deform', 2560)]:
            self.register_buffer(key + '_mean', torch.zeros(dim))
            self.register_buffer(key + '_std', torch.ones(dim))

    def project(self, f6, deform):
        features = []
        if self.f6_projection is not None:
            features.append(self.f6_projection((f6 - self.f6_mean) / self.f6_std))
        if self.deform_projection is not None:
            features.append(self.deform_projection((deform - self.deform_mean) / self.deform_std))
        return torch.cat(features, dim=-1) if len(features) == 2 else features[0]

    def forward(self, f6=None, deform=None, lengths=None, state=None):
        value = self.project(f6, deform)
        if self.head_kind == 'lstm':
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(value, lengths.cpu(), batch_first=True, enforce_sorted=False)
                output, new_state = self.context(packed, state)
                value, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=value.shape[1])
            else:
                value, new_state = self.context(value, state)
        else:
            value, new_state = self.context(value), None
        return self.prediction(value), new_state


# Preserve the original binary experiment import and checkpoint compatibility.
BinaryProbe = FrameProbe


class OnlinePredictor:
    """Consume one right-hand tick at a time; no future samples or bidirectional state."""
    def __init__(self, checkpoint, device='cpu'):
        blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if blob.get('output_unit') in ('interval', 'align_window'):
            raise ValueError('Use the matching IntervalProbe/AlignWindowProbe; OnlinePredictor is frame-wise')
        config = blob['model_config']
        self.device = torch.device(device)
        self.encoders = FrozenEncoders().to(device)
        expected = blob.get('encoder_sha256', {})
        if expected and (sha(self.encoders.f6_path) != expected['f6'] or
                         sha(self.encoders.deform_path) != expected['deform']):
            raise ValueError('Encoder weights differ from the trained feature representation')
        self.failure_threshold = float(blob.get('failure_threshold', 0.5))
        self.probe = BinaryProbe(**config).to(device).eval()
        self.probe.load_state_dict(blob['state_dict'], strict=True)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=16)
        self.state = None
        self.last_tick = None

    @torch.inference_mode()
    def step(self, f6, deform=None, tick=None, valid=True):
        if not valid or (tick is not None and self.last_tick is not None and tick != self.last_tick + 1):
            self.reset()
            if not valid:
                return None
        self.last_tick = tick
        raw = np.asarray(f6, dtype=np.float32)
        if raw.shape != (5, 6) or not np.isfinite(raw).all():
            self.reset()
            return None
        self.history.append(raw.copy())
        if len(self.history) < 16:
            return None
        f6_features = deform_features = None
        if self.probe.f6_projection is not None:
            window = torch.from_numpy(np.stack(self.history)).unsqueeze(0).to(self.device)
            f6_features = self.encoders.f6_features(window).unsqueeze(1)
        if self.probe.deform_projection is not None:
            image = np.asarray(deform, dtype=np.float32)
            if image.shape != (5, 1, 240, 240) or not np.isfinite(image).all():
                self.reset()
                return None
            deform_features = self.encoders.deform_features(torch.from_numpy(image).unsqueeze(0).to(self.device)).unsqueeze(1)
        logits, self.state = self.probe(f6_features, deform_features, state=self.state)
        probabilities = logits.softmax(-1)[0, 0].cpu()
        return probabilities.numpy() if self.probe.num_classes == 3 else float(probabilities[1])
