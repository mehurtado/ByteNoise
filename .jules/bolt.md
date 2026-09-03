## 2024-05-24 - Einops string parsing overhead
**Learning:** In highly called functions like multi-head attention forward passes within this codebase's U-Net architecture, the `einops.rearrange` function introduces significant string parsing and map/lambda overhead when restructuring tensors.
**Action:** Replace `einops.rearrange` with native PyTorch `.view()`, `.transpose()`, and `.contiguous()` in critical training loops like `Attention1D`.

## 2024-05-24 - Precalculate static tensors
**Learning:** `torch.exp(torch.arange...)` calculations in `SinusoidalPosEmb` and `torch.sqrt` calculations in `GaussianDiffusion` are currently recomputed repeatedly but are functionally static for the lifetime of the model.
**Action:** Precompute these values and register them as buffers during object initialization using `self.register_buffer`.
