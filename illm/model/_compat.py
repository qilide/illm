from typing import Dict, List, Optional, Union

from pydantic import (  # noqa: F401
    BaseModel,
    Field,
    EncoderProtocol,
    ValidationError,
    create_model,
    validate_arguments,
    validator,
)

from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)
from openai.types.shared_params.response_format_text import ResponseFormatText
from typing_extensions import Literal

from openai._types import Body


class JSONSchema(BaseModel):
    name: str
    description: Optional[str] = None
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    strict: Optional[bool] = None


class ResponseFormatJSONSchema(BaseModel):
    json_schema: JSONSchema
    type: Literal["json_schema"]


ResponseFormat = Union[
    ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema
]


class CreateChatCompletionOpenAI(BaseModel):
    """
    Comes from source code: https://github.com/openai/openai-python/blob/main/src/openai/types/chat/completion_create_params.py
    """

    messages: List[Dict]
    model: str
    frequency_penalty: Optional[float]
    logit_bias: Optional[Dict[str, int]]
    logprobs: Optional[bool]
    max_completion_tokens: Optional[int]
    max_tokens: Optional[int]
    n: Optional[int]
    parallel_tool_calls: Optional[bool]
    presence_penalty: Optional[float]
    response_format: Optional[ResponseFormat]
    seed: Optional[int]
    service_tier: Optional[Literal["auto", "default"]]
    stop: Union[Optional[str], List[str]]
    temperature: Optional[float]
    tool_choice: Optional[  # type: ignore
        Union[
            Literal["none", "auto", "required"],
        ]
    ]
    top_logprobs: Optional[int]
    top_p: Optional[float]
    extra_body: Optional[Body]
    user: Optional[str]
