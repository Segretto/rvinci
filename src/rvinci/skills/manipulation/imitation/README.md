# Manipulation: Imitation Skill

This skill provides a deep learning fusion network architecture designed to merge DINOv3 visual object queries with proprioceptive robot wrist poses to predict a 7D relative displacement. It serves as the primary neural network "brain" for imitation learning tasks spanning visual grasping and end-effector localization.

### Public API Usage Example

```python
from rvinci.skills.manipulation.imitation.api import PoseFusionHead, ImitationConfig

config = ImitationConfig(output_quaternion=True)
head = PoseFusionHead(
    query_dim=config.query_dim,
    proprio_dim=config.proprio_dim,
    hidden_dim=config.hidden_dim, 
    output_quaternion=config.output_quaternion
)

# ... inside inference loop ...
translation, rotation_quat = head(object_query, current_tcp_pose)
```

_Note: This skill is Hydra-agnostic; orchestration, dataset iteration, and configurations belong in projects._
