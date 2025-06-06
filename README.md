# ByteNoise v0.1  
*A Byte-Level Diffusion Model for Byte Sequence Generation*

**ByteNoise** is an ongoing research project exploring the viability of denoising diffusion probabilistic models (DDPMs) applied directly to byte sequences. The model operates without tokenization, treating text as raw byte data and attempting to reconstruct coherent sequences from random noise.

This repository contains:
- A custom U-Net architecture for 1D byte diffusion
- Training loop and configuration for autoregressive sampling
- Roadmap for improvements and failure analysis

> **Warning:** This model is still unstable. Current outputs are mostly noise. Training is active and this repository will be updated as results evolve.

