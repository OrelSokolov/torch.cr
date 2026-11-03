#!/usr/bin/env python3
"""Dumps the golden oracle tree with python torch; mirrors bin/oracle_dump.cr
byte-for-byte. Output: oracle/golden (compared by twinshot)."""
import json
import pathlib
import sys

import torch

out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "oracle/golden")
out_dir.mkdir(parents=True, exist_ok=True)

fixtures = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
golden = json.loads((fixtures / "golden.json").read_text())


def fmt(values):
    return ["%0.6f" % v for v in values]


def grid_rows(values, width):
    return [" ".join(fmt(row)) for row in torch.tensor(values).reshape(-1, width).tolist()]


def dtype_name(tensor):
    # Match Crystal's Torch::Kind#to_s ("torch.float32" -> "Float32").
    return str(tensor.dtype).split(".")[-1].capitalize()


# -- tensor io ---------------------------------------------------------------
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
(out_dir / "tensor_io.txt").write_text(
    "\n".join([f"shape={'x'.join(map(str, t.shape))}", f"kind={dtype_name(t)}"]
              + grid_rows(t.flatten().tolist(), 2)) + "\n")

ti = torch.tensor([10, 20, 30], dtype=torch.int64)
(out_dir / "tensor_i64.txt").write_text("\n".join(map(str, ti.tolist())) + "\n")

# -- traced module -----------------------------------------------------------
x = torch.tensor(golden["x"])
m = torch.jit.load(str(fixtures / "double.pt")).eval()
(out_dir / "forward_double.txt").write_text(
    "\n".join(grid_rows(m(x).flatten().tolist(), 3)) + "\n")

multi = torch.jit.load(str(fixtures / "multi.pt")).eval()
y, rowsum, ret = multi(x, 3)
(out_dir / "forward_multi.txt").write_text(
    "\n".join(grid_rows(y.flatten().tolist(), 3)
              + fmt(rowsum.tolist()) + [str(ret)]) + "\n")

lin = torch.jit.load(str(fixtures / "linear.pt")).eval()
(out_dir / "forward_linear.txt").write_text(
    "\n".join(grid_rows(lin(x).flatten().tolist(), 2)) + "\n")

names = sorted(n for n, _ in lin.named_parameters())
(out_dir / "linear_params.txt").write_text("\n".join(names) + "\n")

print(f"golden oracle tree dumped to {out_dir}")
