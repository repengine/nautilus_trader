from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.skipif(pytest.importorskip("torch", reason="torch not installed") is None, reason="no torch")
def test_torchscript_export_parity(tmp_path):
    import torch
    import torch.nn as nn

    from ml.training.export import convert_to_torchscript

    class Simple(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 3), nn.SiLU(), nn.Linear(3, 1))

        def forward(self, x):
            return self.net(x)

    model = Simple().eval()
    x = np.random.randn(5, 4).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()

    out_path = tmp_path / "simple"
    ts_path = convert_to_torchscript(model, x[:1], out_path)
    assert ts_path.exists()

    ts = torch.jit.load(str(ts_path))
    with torch.no_grad():
        out = ts(torch.from_numpy(x)).numpy()
    assert np.allclose(out, ref, atol=1e-5)
