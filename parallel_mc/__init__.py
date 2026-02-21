"""
parallel_mc - Parallel Multi-Group Monte Carlo Neutron Transport
for 100 MWth Marine Molten Chloride Fast Reactor (MCFR)

Three parallelization backends:
  - CPU: multiprocessing + Numba JIT
  - CUDA: CuPy + Numba CUDA kernels
  - Metal: Apple Metal GPU compute shaders

8-group fast spectrum energy structure.
Validates against OpenMC reference (k_eff = 0.9327).
"""
__version__ = "0.1.0"
