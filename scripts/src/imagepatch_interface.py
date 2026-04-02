from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Optional


class AbstractImagePatch(ABC):
    @abstractmethod
    def __init__(
        self,
        image,
        left: Optional[int],
        lower: Optional[int],
        right: Optional[int],
        upper: Optional[int],
    ):
        pass

    @abstractmethod
    def find(self, object_name: str) -> list["AbstractImagePatch"]:
        pass

    @abstractmethod
    def exists(self, object_name: str) -> bool:
        pass

    @abstractmethod
    def verify_property(self, object_name: str, visual_property: str) -> bool:
        pass

    @abstractmethod
    def best_text_match(self, option_list: list[str]) -> str:
        pass

    @abstractmethod
    def simple_query(self, question: Optional[str]) -> str:
        pass

    @abstractmethod
    def crop(
        self, left: int, lower: int, right: int, upper: int
    ) -> "AbstractImagePatch":
        pass

    @abstractmethod
    def overlaps_with(self, left: int, lower: int, right: int, upper: int) -> bool:
        pass

    @abstractmethod
    def llm_query(self, question: str, long_answer: bool) -> str:
        pass
