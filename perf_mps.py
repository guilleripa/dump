# Compare with MPS against CPU
import timeit

import torch

tensor_size = 10000

b_mps = torch.rand((tensor_size, tensor_size), device="mps")
b_cpu = torch.rand((tensor_size, tensor_size), device="cpu")

print("mps", timeit.timeit(lambda: b_mps @ b_mps, number=10))
print("cpu", timeit.timeit(lambda: b_cpu @ b_cpu, number=10))
