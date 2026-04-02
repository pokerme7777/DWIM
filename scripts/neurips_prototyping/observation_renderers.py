from abc import ABC, abstractmethod

from .environment import (
    CodeObservation,
    NonCodeObservation,
    NullObservation,
    Observation,
)


class AbstractRenderer(ABC):
    @abstractmethod
    def __call__(self, observation: Observation) -> str:
        pass


class SimpleXmlRenderer(AbstractRenderer):
    def __call__(self, observation: Observation) -> str:
        match observation:
            case CodeObservation():
                # Note: We don't render the program state here, only
                # the result; this is just to save tokens.
                return f"<result>{observation.execution_result}</result>"
            case NonCodeObservation():
                return f"<result>{observation.content}</result>"
            case NullObservation():
                return "<result></result>"
            case _:
                raise ValueError(f"Unknown observation type: {observation}")
