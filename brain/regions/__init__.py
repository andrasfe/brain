"""Brain regions, each a specialized agent."""
from .sensory_cortex import SensoryCortex
from .amygdala import Amygdala
from .hippocampus import Hippocampus
from .prefrontal import Prefrontal
from .basal_ganglia import BasalGanglia
from .broca import Broca
from .interoception import Interoception
from .default_mode import DefaultMode
from .locus_coeruleus import LocusCoeruleus
from .vta import VTA
from .cerebellum import Cerebellum, CerebellumPrediction
from .predictor import Predictor, PlanEvaluation
from .motor_cortex import MotorCortex, has_motor_intent
from .occipital import Occipital

__all__ = [
    "SensoryCortex",
    "Amygdala",
    "Hippocampus",
    "Prefrontal",
    "BasalGanglia",
    "Broca",
    "Interoception",
    "DefaultMode",
    "LocusCoeruleus",
    "VTA",
    "Cerebellum",
    "CerebellumPrediction",
    "Predictor",
    "PlanEvaluation",
    "MotorCortex",
    "has_motor_intent",
    "Occipital",
]
