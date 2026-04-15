"""
MAIA MUSE — Detector Wrapper
Wraps the fakeprint-based ONNX inference for use in the Streamlit demo.
"""

import warnings
import tempfile
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import librosa
from scipy.ndimage import minimum_filter1d
import onnxruntime as ort
import yaml

warnings.filterwarnings("ignore")

CONFIG_PATH = Path(__file__).parent / "config.yaml"
MODEL_PATH  = Path(__file__).parent / "models" / "ai_music_detector.onnx"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class MUSEDetector:
    """
    Fakeprint-based AI music detector.
    Detects periodic spectral artifacts left by Suno/Udio deconvolution layers.
    """

    def __init__(self):
        cfg = load_config()
        audio_cfg = cfg["audio"]
        fp_cfg    = cfg["fakeprint"]

        self.sample_rate = audio_cfg["sample_rate"]
        self.n_fft       = audio_cfg["n_fft"]
        self.max_duration = audio_cfg["max_duration"]
        self.freq_min    = fp_cfg["freq_min"]
        self.freq_max    = fp_cfg["freq_max"]
        self.hull_area   = fp_cfg["hull_area"]
        self.max_db      = fp_cfg["max_db"]
        self.min_db      = fp_cfg["min_db"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, power=2, normalized=False
        ).to(self.device)

        freq_bins = np.linspace(0, self.sample_rate / 2, (self.n_fft // 2) + 1)
        self.freq_mask  = (freq_bins >= self.freq_min) & (freq_bins <= self.freq_max)
        self.freq_range = freq_bins[self.freq_mask]

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}\n"
                "Copy ai_music_detector.onnx from ai-music-detector/src/models/ into models/"
            )

        self.session    = ort.InferenceSession(str(MODEL_PATH))
        self.input_name = self.session.get_inputs()[0].name
        self.n_features = self.session.get_inputs()[0].shape[1]

    def load_audio(self, file_path: str) -> torch.Tensor:
        waveform, _ = librosa.load(file_path, sr=self.sample_rate, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        audio = torch.from_numpy(waveform).float()
        max_s = self.max_duration * self.sample_rate
        if audio.shape[1] > max_s:
            audio = audio[:, :max_s]
        return audio

    def load_audio_bytes(self, data: bytes, suffix: str = ".mp3") -> torch.Tensor:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            return self.load_audio(tmp_path)
        finally:
            os.unlink(tmp_path)

    def get_waveform(self, audio: torch.Tensor) -> np.ndarray:
        """Return mono waveform as numpy array for visualization."""
        wav = audio.mean(dim=0) if audio.shape[0] > 1 else audio[0]
        return wav.cpu().numpy()

    def compute_fakeprint(self, audio: torch.Tensor) -> np.ndarray:
        audio = audio.to(self.device)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        with torch.no_grad():
            spec = self.stft(audio)

        spec_db = 10 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
        mean_spec = spec_db.mean(dim=(0, 2)).cpu().numpy()
        freq_spec = mean_spec[self.freq_mask]

        hull = minimum_filter1d(freq_spec, size=self.hull_area, mode="nearest")
        hull = np.clip(hull, self.min_db, None)

        residue = np.clip(freq_spec - hull, 0, None)
        residue = np.clip(residue, 0, self.max_db)
        max_val = np.max(residue) + 1e-6

        return (residue / max_val).astype(np.float32)

    def predict(self, audio: torch.Tensor) -> dict:
        fakeprint = self.compute_fakeprint(audio)

        if len(fakeprint) != self.n_features:
            old_x = np.linspace(0, 1, len(fakeprint))
            new_x = np.linspace(0, 1, self.n_features)
            fakeprint = np.interp(new_x, old_x, fakeprint).astype(np.float32)

        out = self.session.run(None, {self.input_name: fakeprint.reshape(1, -1)})
        prob = float(out[0][0, 0])

        return {
            "probability":  prob,
            "is_ai":        prob >= 0.5,
            "label":        "AI-Generated" if prob >= 0.5 else "Authentic",
            "confidence":   abs(prob - 0.5) * 2,
            "fakeprint":    fakeprint,
        }
