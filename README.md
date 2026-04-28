# Partition-of-Unity Gaussian Kolmogorov-Arnold Networks

Main source code for the paper:

**Partition-of-Unity Gaussian Kolmogorov--Arnold Networks**  
Amir Noorizadegan  
Department of Mathematics, Hong Kong Baptist University  
arXiv:2604.23599, 2026  
DOI: https://doi.org/10.48550/arXiv.2604.23599

## Overview

This repository contains the main model implementations for RBF-based KANs, with a focus on the partition-of-unity Gaussian KAN (PU-GKAN).

The implemented models are:

- `GKAN`: Gaussian KAN
- `PU-GKAN`: partition-of-unity Gaussian KAN
- `MaternKAN`: Matérn KAN
- `PU-MaternKAN`: partition-of-unity Matérn KAN

In PU-GKAN, the Gaussian basis values on each edge are normalized by their local sum over fixed centers. This gives a Shepard-type partition-of-unity basis while keeping the standard edge-based KAN architecture.

## Basic setting

The experiments in the paper use:

- inputs mapped to `[0, 1]^d`
- uniformly spaced RBF centers
- Halton points for training and validation
- AdamW optimizer
- float32 precision
- architectures such as `(2, 12, 12, 1)`
- Gaussian and Matérn RBF bases
- fixed scale parameter `epsilon`

## Model files

```text
model_gkan.py
model_pu_gkan.py
model_maternkan.py
model_pu_maternkan.py


@article{Noorizadegan2026PUGKAN,
  title  = {Partition-of-Unity Gaussian Kolmogorov--Arnold Networks},
  author = {Noorizadegan, Amir},
  journal = {arXiv preprint arXiv:2604.23599},
  year   = {2026},
  doi    = {10.48550/arXiv.2604.23599}
}

