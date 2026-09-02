import torch
from main import UNet1D
import time

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}")

    # Initialize model using config from main.py
    model = UNet1D(model_dim=192).to(device)

    # Dummy inputs simulating a forward pass in the diffusion loop
    # shape: (batch_size, seq_len, in_embed_dim)
    x = torch.randn(16, 256, 128).to(device)

    # shape: (batch_size,)
    t = torch.randint(0, 1000, (16,)).to(device)

    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.time()

    # Run a few dummy forward/backward passes to simulate training step
    for i in range(10):
        out = model(x, t)
        loss = out.mean()
        loss.backward()

    end_time = time.time()

    print(f"Time taken for 10 forward/backward passes: {end_time - start_time:.4f} seconds")

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak VRAM allocated: {peak_memory:.2f} MB")
    else:
        print("Run on a CUDA-enabled device to measure VRAM usage.")

if __name__ == "__main__":
    benchmark()
