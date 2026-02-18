from typing import Type, TypeVar, Any
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def validate_config(cfg: Any, schema: Type[T]) -> T:
    """
    Validate a configuration object (DictConfig or dict) against a Pydantic schema.
    
    Args:
        cfg: The configuration object (usually from Hydra/OmegaConf).
        schema: The Pydantic model class to validate against.
        
    Returns:
        The validated Pydantic model instance.
        
    Raises:
        ValidationError: If validation fails.
    """
    if isinstance(cfg, DictConfig):
        # Resolve interpolations before validation
        OmegaConf.resolve(cfg)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        raise ValueError(f"Unsupported config type: {type(cfg)}")
    
    try:
        return schema.model_validate(cfg_dict)
    except ValidationError as e:
        # Re-raising with a clear message might be helpful, but Pydantic's error is usually good.
        raise e
