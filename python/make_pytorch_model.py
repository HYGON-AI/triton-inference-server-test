"""Create a tiny TorchScript model for the Triton PyTorch backend test."""
import sys

import torch


class Add(torch.nn.Module):
    def forward(self, a, b):
        return a + b


if __name__ == "__main__":
    out = sys.argv[1]
    model = torch.jit.script(Add())
    model.eval()
    torch.jit.save(model, out)
    print("saved", out)
