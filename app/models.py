from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    content: str
    assertions: list[str]


class ClassifyResponse(BaseModel):
    results: dict[str, float]


class ExtractRequest(BaseModel):
    url: str


class ExtractResponse(BaseModel):
    content: str
    title: str
