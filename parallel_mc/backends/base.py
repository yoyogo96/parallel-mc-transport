"""Abstract base class for MC transport backends."""
from abc import ABC, abstractmethod
from ..particle import ParticleBank, FissionBank
from ..geometry import MCFRGeometry
from ..materials import Material
from ..tallies import TallyAccumulator
from typing import Dict, Optional
import numpy as np


class MCBackend(ABC):
    """Abstract interface for Monte Carlo transport backends.

    Each backend implements particle-level parallelism within a single batch.
    The eigenvalue driver (PowerIteration) calls transport_batch() once per generation.
    """

    @abstractmethod
    def transport_batch(
        self,
        source_bank: ParticleBank,
        geometry: MCFRGeometry,
        materials: Dict[int, Material],
        tallies: Optional[TallyAccumulator],
        rng: np.random.Generator,
    ) -> FissionBank:
        """Transport all particles in source_bank through one generation.

        Args:
            source_bank: ParticleBank with source particles for this batch
            geometry: MCFRGeometry describing the core+reflector
            materials: dict mapping region_id to Material
            tallies: TallyAccumulator for scoring (None for inactive batches)
            rng: numpy random Generator for reproducibility

        Returns:
            FissionBank containing new fission site positions and groups
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable backend name, e.g. 'CPU (8 cores)', 'Metal (M2 GPU)'."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's hardware/libraries are available."""
        pass
