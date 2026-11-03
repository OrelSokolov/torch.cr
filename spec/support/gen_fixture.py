#!/usr/bin/env python3
"""Regenerates TorchScript fixtures + golden values used by specs and the
twinshot oracle tree. Run with a python that has torch (see Makefile PY)."""
import json
import pathlib

import torch

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)


class Double(torch.nn.Module):
    def forward(self, x):
        return x * 2


class Multi(torch.nn.Module):
    """Tensor + int args, tuple(int tensor, tensor, int) result."""

    def forward(self, x, scale: int):
        y = x * scale
        return (y, y.sum(dim=-1), scale * 2)


class Linear3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(2.0)

    def forward(self, x):
        return self.lin(x)


x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

torch.jit.trace(Double(), torch.zeros(2, 3)).save(str(FIXTURES / "double.pt"))
torch.jit.script(Multi()).save(str(FIXTURES / "multi.pt"))
torch.jit.trace(Linear3(), torch.zeros(1, 3)).save(str(FIXTURES / "linear.pt"))

golden = {
    "x": x.tolist(),
    "double": (x * 2).tolist(),
    "multi_scale": 3,
    "multi_y": (x * 3).tolist(),
    "multi_rowsum": (x * 3).sum(dim=-1).tolist(),
    "multi_ret_int": 6,
    "linear": (x @ torch.full((3, 2), 2.0)).tolist(),
    "linear_weight": [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
}
(FIXTURES / "golden.json").write_text(json.dumps(golden, indent=2))
print("fixtures written to", FIXTURES)
