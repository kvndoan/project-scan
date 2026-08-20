from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MappingStats:
    keyframes: int = 0
    landmarks: int = 0
    dense_points: int = 0


@dataclass(frozen=True)
class SlamUpdate:
    timestamp: float
    tracking_status: str
    camera_to_world: np.ndarray
    mapping: MappingStats


class SlamBackend(ABC):
    def __enter__(self) -> "SlamBackend":
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.stop()

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def wait_for_update(self) -> SlamUpdate:
        pass