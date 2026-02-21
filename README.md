# Parallel Monte Carlo Neutron Transport for MCFR

**용융염화물 고속로를 위한 병렬 몬테카를로 중성자 수송 코드**

> **Note: 100% AI-Generated**
> This entire codebase -- including all Monte Carlo transport algorithms, nuclear data processing, GPU kernels, and validation infrastructure -- was generated entirely by AI (Claude, Anthropic). No human-written code is included.

## Overview

A multigroup Monte Carlo neutron transport code designed for Molten Chloride Fast Reactor (MCFR) eigenvalue calculations. Supports three compute backends for hardware-adaptive performance:

- **Pure Python / NumPy** -- portable baseline
- **Numba JIT + multicore CPU** -- parallel CPU acceleration
- **Apple Metal GPU** -- GPU-accelerated transport on macOS

## Features

- 8-group energy structure (10 MeV to thermal) optimized for fast-spectrum reactors
- Power iteration eigenvalue solver with Shannon entropy convergence diagnostics
- Region-wise flux tallies with batch statistics
- NaCl-KCl-UCl3 fuel salt and BeO/SS316H reflector materials
- Validation framework against OpenMC continuous-energy reference
- CLI with preset run modes (quick / production)

## Installation

```bash
# Basic (pure Python, no JIT)
pip install .

# With Numba CPU acceleration
pip install ".[cpu-jit]"

# With Apple Metal GPU support
pip install ".[metal]"

# Development (includes pytest)
pip install ".[dev]"
```

## Quick Start

```bash
# Quick test run (10k particles, 100 batches)
parallel-mc --quick

# Production run (50k particles, 300 batches)
parallel-mc --production

# Specify backend and parameters
parallel-mc --backend cpu --particles 20000 --batches 200

# Metal GPU backend
parallel-mc --backend metal --particles 50000

# List available backends
parallel-mc --list-backends

# Validate against OpenMC reference
parallel-mc --validate
```

Or run as a module:

```bash
python -m parallel_mc --quick
python -m parallel_mc --validate
```

## Performance

| Backend | Throughput | Speedup | Hardware |
|---------|-----------|---------|----------|
| Pure Python | 451 particles/s | 1x | Single core |
| Numba + 4 cores | 3,198 particles/s | 7.1x | CPU (4-core) |
| Apple Metal GPU | 1,400,000 particles/s | 3,100x | M-series GPU |

## Architecture

```
parallel_mc/
  constants.py       8-group energy boundaries and physical constants
  nuclear_data.py    Microscopic cross-section library (ENDF/B-VIII.0 based)
  materials.py       Macroscopic cross-section builder (fuel salt, reflectors)
  geometry.py        MCFR cylindrical geometry with region tracking
  particle.py        Neutron state and transport mechanics
  physics.py         Collision physics (scatter, absorption, fission)
  tallies.py         Region/group flux and reaction rate tallies
  entropy.py         Shannon entropy for fission source convergence
  eigenvalue.py      Power iteration k-eigenvalue solver
  cli.py             Command-line interface
  backends/
    base.py          Abstract backend interface
    cpu.py           NumPy + optional Numba JIT backend
    cuda.py          NVIDIA CUDA backend (via Numba CUDA)
    metal.py         Apple Metal GPU backend (via metalcompute)
  validation/
    compare.py       Automated comparison against OpenMC reference
```

## Validation

The code includes built-in validation against OpenMC (continuous-energy Monte Carlo). Expected agreement is within 2-5% for k-effective, with differences attributable to the 8-group energy discretization.

```bash
parallel-mc --validate
```

Reference: k_eff = 0.9327 +/- 0.00035 (OpenMC, 4M active histories)

## License

MIT License. See [LICENSE](LICENSE) for details.
