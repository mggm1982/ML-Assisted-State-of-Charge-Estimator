"""Feedforward residual model, feature scaling and causal training windows."""

from __future__ import annotations
import numpy as np


def network(kind, n_features, torch):
    """Construct the 60-48-24-2 feedforward network used by the saved weights."""
    nn = torch.nn
    if kind == "fnn":

        class FNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(n_features * 4, 48),
                    nn.ReLU(),
                    nn.Linear(48, 24),
                    nn.ReLU(),
                    nn.Linear(24, 2),
                    nn.Tanh(),
                )

            def forward(self, x):
                q = torch.cat(
                    (x[:, -1], x[:, -8:].mean(1), x[:, -32:].mean(1), x.mean(1)), dim=-1
                )
                return self.model(q)

        return FNN()
    raise ValueError("Supported network architecture: fnn")


class FeatureScaler:
    """Standardize features using statistics fitted to the training recordings."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, training_groups):
        a = np.concatenate([g["X"][::10] for g in training_groups], axis=0)
        self.mean = np.mean(a, axis=0).astype(np.float32)
        self.std = np.maximum(np.std(a, axis=0), np.float32(1.0e-4))
        return self

    def transform(self, a):
        return np.clip(
            (np.asarray(a, np.float32) - self.mean) / self.std, -8.0, 8.0
        ).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, x):
        s = cls()
        s.mean = np.asarray(x["mean"], np.float32)
        s.std = np.asarray(x["std"], np.float32)
        return s


class WindowDataset:
    """Build fixed-length causal windows without crossing recording boundaries."""

    def __init__(self, groups, scaler, window=64, stride=8, burn=60):
        self.X = []
        self.Y = []
        self.ids = []
        self.window = window
        for group in groups:
            X = scaler.transform(group["X"])
            Y = np.asarray(group["Y"], np.float32)
            ii = np.arange(min(burn, len(X) - 1), len(X), stride, dtype=int)
            if not len(ii):
                continue
            ix = len(self.X)
            self.X.append(X)
            self.Y.append(Y)
            self.ids.extend((ix, int(k)) for k in ii)
        if not self.ids:
            raise ValueError("No eligible training windows")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch

        g, k = self.ids[i]
        x = self.X[g]
        lo = max(0, k + 1 - self.window)
        w = x[lo : k + 1]
        if len(w) < self.window:
            w = np.concatenate(
                (np.repeat(w[:1], self.window - len(w), axis=0), w), axis=0
            )
        return torch.from_numpy(w.copy()), torch.from_numpy(self.Y[g][k].copy())


def load_artifact(path):
    """Load network weights, feature scaling and correction settings."""
    import json
    import torch
    from pathlib import Path

    path = Path(path)
    cfg = json.loads(path.with_suffix(".json").read_text())
    model = network(cfg["kind"], cfg["features"], torch)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model, FeatureScaler.from_dict(cfg["scaler"]), cfg
