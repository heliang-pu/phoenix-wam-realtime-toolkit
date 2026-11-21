"""What this GPU reaches on the trunk's own matrix products (bf16, 576 tokens), against what the MoT forward takes."""
import time

import torch

dev = "cuda"
print("gpu", torch.cuda.get_device_name(0), torch.__version__)


def bench(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


for tokens in (576, 384, 2304):
    x = torch.randn(1, tokens, 3072, device=dev, dtype=torch.bfloat16)
    h = torch.randn(1, tokens, 14336, device=dev, dtype=torch.bfloat16)
    sq = torch.nn.Linear(3072, 3072, device=dev, dtype=torch.bfloat16)
    up = torch.nn.Linear(3072, 14336, device=dev, dtype=torch.bfloat16)
    down = torch.nn.Linear(14336, 3072, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        for name, fn, flops in (("3072x3072", lambda: sq(x), 2 * tokens * 3072 * 3072),
                                ("3072x14336", lambda: up(x), 2 * tokens * 3072 * 14336),
                                ("14336x3072", lambda: down(h), 2 * tokens * 3072 * 14336)):
            t = bench(fn)
            print(f"tokens {tokens:5d} linear {name:11s} {t * 1e6:8.1f} us  {flops / t / 1e12:6.1f} TFLOPS")
        # one layer's linears back to back: 8 square (self + cross attention q k v o) and the feed-forward pair
        def layer():
            y = x
            for _ in range(8):
                y = sq(y)
            return down(up(y))
        t = bench(layer, 100)
        flops = 2 * tokens * (8 * 3072 * 3072 + 2 * 3072 * 14336)
        print(f"tokens {tokens:5d} one layer's 10 linears {t * 1e3:6.3f} ms  {flops / t / 1e12:6.1f} TFLOPS  -> 30 layers {30 * t * 1e3:6.1f} ms")
