from typing import Annotated
from pydantic import Field
from pydantic import BaseModel

class DispatcherTask(BaseModel):
    order_number: Annotated[int, Field(gt=0)]
    description: Annotated[str, Field()]