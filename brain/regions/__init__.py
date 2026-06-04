"""Brain regions, each a specialized agent."""
from .sensory_cortex import SensoryCortex
from .amygdala import Amygdala
from .hippocampus import Hippocampus
from .prefrontal import Prefrontal
from .basal_ganglia import BasalGanglia
from .broca import Broca

__all__ = [
    "SensoryCortex",
    "Amygdala",
    "Hippocampus",
    "Prefrontal",
    "BasalGanglia",
    "Broca",
]
