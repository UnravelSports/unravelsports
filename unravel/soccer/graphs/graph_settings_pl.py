from dataclasses import dataclass, asdict

from ...utils import DefaultGraphSettings
from enum import Enum

from dataclasses import dataclass, field
from kloppy.domain import MetricPitchDimensions

from ..dataset import Constant


@dataclass
class GraphSettingsPolars(DefaultGraphSettings):
    ball_id: str = Constant.BALL
    goalkeeper_id: str = "GK"
    boundary_correction: float = None
    non_potential_receiver_node_value: float = 0.1
    ball_carrier_treshold: float = 25.0
    pitch_dimensions: MetricPitchDimensions = field(
        init=False, repr=False, default_factory=MetricPitchDimensions
    )

    def __post_init__(self):
        self._sport_specific_checks()

    @property
    def pitch_dimensions(self) -> int:
        return self._pitch_dimensions

    @pitch_dimensions.setter
    def pitch_dimensions(self, pitch_dimensions: MetricPitchDimensions) -> None:
        self._pitch_dimensions = pitch_dimensions

    def _sport_specific_checks(self):
        if self.non_potential_receiver_node_value > 1:
            self.non_potential_receiver_node_value = 1
        elif self.non_potential_receiver_node_value < 0:
            self.non_potential_receiver_node_value = 0

    def to_dict(self):
        """Custom serialization method that skips Enum fields (like 'unit') and serializes others."""
        
        def make_serializable(obj):
            if isinstance(obj, Enum):
                return None
            elif isinstance(obj, (int, float, str, bool, type(None), list, dict)):
                return obj
            elif isinstance(obj, MetricPitchDimensions):  
                return {
                    key: make_serializable(value)
                    for key, value in obj.__dict__.items()
                    if not isinstance(value, Enum)
                }
            elif hasattr(obj, "__dict__"):  
                return {key: make_serializable(value) for key, value in obj.__dict__.items()}
            return None

        return {key: make_serializable(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data):
        """Custom deserialization method"""
        if "pitch_dimensions" in data:
            data["pitch_dimensions"] = MetricPitchDimensions(**data["pitch_dimensions"])
        return cls(**data)
