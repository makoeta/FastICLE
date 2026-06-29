from typing import Annotated

from pydantic import BaseModel, Field


class TrainingTaskList(BaseModel):
    tasks: Annotated[list[str], Field(min_length=1)]
