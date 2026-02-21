"""
Backend registry with automatic detection.

Priority order: Metal (Apple GPU) > CUDA (NVIDIA GPU) > CPU (always available)
"""
from .base import MCBackend
from .cpu import CPUBackend


def list_backends():
    """List all available backends with their status."""
    backends = []

    # CPU is always available
    cpu = CPUBackend()
    backends.append(('CPU', cpu.get_name(), True))

    # Try CUDA
    try:
        from .cuda import CUDABackend
        cuda = CUDABackend()
        backends.append(('CUDA', cuda.get_name(), cuda.is_available()))
    except ImportError:
        backends.append(('CUDA', 'Not installed (pip install cupy-cuda12x)', False))

    # Try Metal
    try:
        from .metal import MetalBackend
        metal = MetalBackend()
        backends.append(('Metal', metal.get_name(), metal.is_available()))
    except ImportError:
        backends.append(('Metal', 'Not installed (pip install metalcompute)', False))

    return backends


def auto_select_backend() -> MCBackend:
    """Auto-select the best available backend.

    Priority: Metal > CUDA > CPU
    """
    # Try Metal first (Apple Silicon)
    try:
        from .metal import MetalBackend
        metal = MetalBackend()
        if metal.is_available():
            return metal
    except (ImportError, Exception):
        pass

    # Try CUDA
    try:
        from .cuda import CUDABackend
        cuda = CUDABackend()
        if cuda.is_available():
            return cuda
    except (ImportError, Exception):
        pass

    # Fallback to CPU (always available)
    return CPUBackend()


def get_backend(name: str) -> MCBackend:
    """Get a specific backend by name.

    Args:
        name: 'cpu', 'cuda', or 'metal'

    Returns:
        MCBackend instance

    Raises:
        ValueError if backend not available
    """
    name = name.lower()

    if name == 'cpu':
        return CPUBackend()
    elif name == 'cuda':
        from .cuda import CUDABackend
        backend = CUDABackend()
        if not backend.is_available():
            raise ValueError("CUDA backend not available (no GPU or CuPy not installed)")
        return backend
    elif name == 'metal':
        from .metal import MetalBackend
        backend = MetalBackend()
        if not backend.is_available():
            raise ValueError("Metal backend not available (not macOS or metalcompute not installed)")
        return backend
    else:
        raise ValueError(f"Unknown backend: {name}. Choose from: cpu, cuda, metal")
