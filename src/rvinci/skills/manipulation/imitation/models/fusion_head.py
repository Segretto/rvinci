import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseFusionHead(nn.Module):
    def __init__(
        self, query_dim=256, proprio_dim=7, hidden_dim=256, output_quaternion=True
    ):
        super(PoseFusionHead, self).__init__()
        self.output_quaternion = output_quaternion

        # 1. Proprioceptive Encoder
        # Upscales the 7D wrist_frame pose to prevent it from being
        # overwhelmed by the high-dimensional visual query.
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64), nn.GELU(), nn.Linear(64, 128), nn.LayerNorm(128)
        )

        # 2. Multimodal Fusion
        # Fuses the visual query (DINOv3/Mask2Former) with the encoded TCP state
        fusion_input_dim = query_dim + 128
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
        )

        # 3. Output Heads
        # Translation: Outputs delta x, y, z
        self.translation_head = nn.Linear(hidden_dim // 2, 3)

        # Rotation: Outputs either 6D continuous representation or 4D unnormalized quaternion
        rot_out_dim = 4 if self.output_quaternion else 6
        self.rotation_head = nn.Linear(hidden_dim // 2, rot_out_dim)

    def forward(self, object_query, tcp_pose):
        """
        Args:
            object_query: Tensor of shape (Batch, query_dim)
            tcp_pose: Tensor of shape (Batch, 7) representing the wrist frame (XYZ + Quat)
        Returns:
            translation: Tensor of shape (Batch, 3) representing predicted translation
            rotation: Tensor of shape (Batch, 4) or (Batch, 6), normalized if quaternion.
        """
        # Encode proprioception
        p_feat = self.proprio_encoder(tcp_pose)

        # Concatenate along the feature dimension
        fused_features = torch.cat([object_query, p_feat], dim=1)

        # Process fused representation
        x = self.fusion_mlp(fused_features)

        # Predict relative pose
        translation = self.translation_head(x)
        rotation_raw = self.rotation_head(x)

        if self.output_quaternion:
            # Normalize to valid SO(3) quaternion space
            rotation = F.normalize(rotation_raw, p=2, dim=-1)
        else:
            rotation = rotation_raw

        return translation, rotation
