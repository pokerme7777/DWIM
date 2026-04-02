from pydantic import BaseModel
from typing import NewType, Union, Literal, Optional

QuestionId = NewType("QuestionId", str)


class VqaRecord(BaseModel):
    question_id: QuestionId
    question: str
    answer: list[str]
    image: str
    dataset: str
    original_question_id: Optional[QuestionId]


class VqaIntrospectSubQaRecord(BaseModel):
    sub_question: str
    sub_answer: str


class VqaIntrospectRecord(BaseModel):
    sub_qa: list[VqaIntrospectSubQaRecord]
    pred_q_type: Union[Literal["perception"], Literal["reasoning"], Literal["invalid"]]


class VqaIntrospectRecordWrapper(BaseModel):
    reasoning_answer_most_common: str
    reasoning_question: str
    image_id: int
    introspect: list[VqaIntrospectRecord]


VqaIntrospectType = dict[QuestionId, VqaIntrospectRecordWrapper]
