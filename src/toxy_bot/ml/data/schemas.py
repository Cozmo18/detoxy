from pydantic import BaseModel, Field


class CommentData(BaseModel):
    id: str
    comment_text: str
    toxic: int = Field(ge=-1, le=1)
    severe_toxic: int = Field(ge=-1, le=1)
    obscene: int = Field(ge=-1, le=1)
    threat: int = Field(ge=-1, le=1)
    insult: int = Field(ge=-1, le=1)
    identity_hate: int = Field(ge=-1, le=1)
