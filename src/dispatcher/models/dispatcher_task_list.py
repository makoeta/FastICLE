from research.icle.src.dispatcher.models.dispatcher_task import DispatcherTask
from typing import Annotated
from pydantic import Field
from pydantic import BaseModel


class DispatcherTasks(BaseModel):
    tasks: Field(list[DispatcherTask])
    