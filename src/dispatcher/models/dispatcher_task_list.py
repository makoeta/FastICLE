from dispatcher.models.dispatcher_task import DispatcherTask
from typing import Annotated
from pydantic import Field
from pydantic import BaseModel


class DispatcherTaskList(BaseModel):
    tasks: Annotated[list[DispatcherTask], Field(min_length=1)]
    