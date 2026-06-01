from fasticrl.models.agent_save_state import AgentSaveState
from typing import Annotated
from pydantic import Field
import yaml

YAML_SUFFIX = ".yaml"
class AgentConfig(AgentSaveState):
    name: Annotated[str, Field()]
    
    def to_yaml(self, path: str):
        
        if not path.endswith(YAML_SUFFIX):
            path += YAML_SUFFIX
        
        with open(path, "w") as outfile:
            yaml.dump(self.model_dump(mode="json"), outfile, default_flow_style=False)
        
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as file:
            data = yaml.safe_load(path)
            return AgentConfig.model_validate(data)
    