from pydantic import BaseModel, Field
from typing import List, Optional

class ProjectConfig(BaseModel):
    # Depending on what the scripts do, we might need a very flexible config for now
    # or just placeholder to pass validation.
    # Since we are consolidating many scripts, we might use subcommands or a 'mode' field.
    
    # For now, let's just allow anything under 'project' and 'skills'
    # extra="allow" is useful for legacy migration where we don't know all fields yet.
    
    class Config:
        extra = "allow"

    project: dict = Field(default_factory=dict)
