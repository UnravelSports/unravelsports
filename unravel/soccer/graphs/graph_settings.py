from dataclasses import dataclass

from ...utils import DefaultGraphSettings

from kloppy.domain import MetricPitchDimensions


@dataclass
class GraphSettings(DefaultGraphSettings):
    infer_goalkeepers: bool = True
    boundary_correction: float = None

    @property
    def pitch_dimensions(self) -> int:
        return self._pitch_dimensions

    @pitch_dimensions.setter
    def pitch_dimensions(self, pitch_dimensions: MetricPitchDimensions) -> None:
        self._pitch_dimensions = pitch_dimensions
