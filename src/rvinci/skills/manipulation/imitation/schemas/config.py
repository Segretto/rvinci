from pydantic import BaseModel, Field


class ImitationConfig(BaseModel):
    """Configuration for the Imitation skill models."""

    query_dim: int = Field(
        256, description="Dimension of the object query (e.g., from Mask2Former)"
    )
    proprio_dim: int = Field(
        7, description="Dimension of the proprioceptive input (e.g., 3D pos + 4D quat)"
    )
    hidden_dim: int = Field(256, description="Hidden dimension size for the fusion MLP")
    output_quaternion: bool = Field(
        True,
        description="Whether to output a 4D quaternion instead of a 6D continuous rotation",
    )

    model_config = {"extra": "forbid"}
