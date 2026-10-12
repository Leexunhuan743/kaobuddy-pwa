from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# 会话 id 会被转发成 HTTP 头，所以限制成不含换行的短标识。
SessionId = Annotated[Optional[str], Field(max_length=120, pattern=r"^[A-Za-z0-9._:+-]+$")]


class ApiConfig(BaseModel):
    provider_name: str = Field(min_length=1, max_length=60)
    base_url: str = Field(min_length=8)
    api_key: str = Field(min_length=1)
    model: str = Field(min_length=1, max_length=120)
    temperature: float = Field(default=0.4, ge=0, le=2)
    max_tokens: int = Field(default=1800, ge=128, le=12000)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("https://", "http://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        return cleaned


class StudyProject(BaseModel):
    subject: str
    exam_date: str
    daily_minutes: int = Field(ge=10, le=1440)
    target_score: Optional[str] = None
    weak_points: Optional[str] = None


class MaterialSummary(BaseModel):
    id: Optional[str] = None
    title: str
    kind: Literal["text", "file", "handwriting", "video", "pdf", "markdown", "document"]
    content: str = ""


class AiRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    api_config: Optional[ApiConfig] = None
    invite_code: Optional[str] = Field(default=None, alias="inviteCode")
    session_id: SessionId = None
    project: StudyProject
    materials: List[MaterialSummary] = []
    extra: Optional[str] = None


class PracticeAnswer(BaseModel):
    question: str
    answer: str


class PracticeRequest(AiRequest):
    answers: List[PracticeAnswer] = []


class ModulePracticeRequest(AiRequest):
    module_title: str = Field(min_length=1, max_length=120)
    exam_points: str = ""


class MockGradeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    api_config: Optional[ApiConfig] = None
    invite_code: Optional[str] = Field(default=None, alias="inviteCode")
    session_id: SessionId = None
    project: StudyProject
    materials: List[MaterialSummary] = []
    exam_content: str = Field(min_length=1)
    user_answers: str = Field(min_length=1)


class DailyPlanModule(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    estimated_minutes: int = Field(default=45, ge=1, le=1440)
    difficulty: Optional[Literal["low", "medium", "high"]] = None
    importance_rank: Optional[int] = None
    exam_points: str = ""
    source_title: str = ""
    evidence: str = ""
    module_status: Literal["todo", "doing"] = "todo"


class DailyPlanRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    api_config: Optional[ApiConfig] = None
    invite_code: Optional[str] = Field(default=None, alias="inviteCode")
    session_id: SessionId = None
    project: StudyProject
    modules: List[DailyPlanModule] = []
    extra: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Any


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    api_config: Optional[ApiConfig] = None
    invite_code: Optional[str] = Field(default=None, alias="inviteCode")
    session_id: SessionId = None
    messages: List[ChatMessage]


class HandwritingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    api_config: Optional[ApiConfig] = None
    invite_code: Optional[str] = Field(default=None, alias="inviteCode")
    session_id: SessionId = None
    image_data_urls: List[str] = Field(min_length=1, max_length=6)
    note_hint: Optional[str] = None


class VideoImportRequest(BaseModel):
    url: HttpUrl


class VideoImportResponse(BaseModel):
    title: str
    description: str = ""
    subtitles: str = ""
    source_url: str
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}


class InviteVerifyRequest(BaseModel):
    code: str = Field(min_length=1, max_length=80)


class InviteVerifyResponse(BaseModel):
    valid: bool
    remaining: int
    remaining_budget_cny: float = Field(alias="remainingBudgetCny")
    message: str


class ChatProxyRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    invite_code: str = Field(min_length=1, max_length=80, alias="inviteCode")
    messages: List[ChatMessage]
    model: Optional[str] = None


class AiResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: str
    remaining: Optional[int] = None
    remaining_budget_cny: Optional[float] = Field(default=None, alias="remainingBudgetCny")
