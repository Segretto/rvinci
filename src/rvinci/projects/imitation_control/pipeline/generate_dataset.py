import os
import cv2
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor
from transformers.utils import ModelOutput
import numpy as np

from rvinci.core.logging import get_logger
from rvinci.skills.perception.dinov3.api import build_dinov3
from rvinci.skills.manipulation.imitation.api import ImitationDataset
from rvinci.libs.vision_data.processing import image_to_tensor
from rvinci.libs.vision_data.constants import IMG_MEAN, IMG_STD

log = get_logger(__name__)


def generate_pipeline(config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # 1. Setup Model
    log.info("Loading DINOv3 Backbone...")
    # Load custom finetuned dinov3 architecture from local repository
    dino_dir = os.path.join(os.getcwd(), "modules/dinov3")
    weights_path = getattr(
        config.skills.dinov3, "weights_path", "models/dinov3_vits16plus.pth"
    )
    weights_full_path = os.path.join(os.getcwd(), weights_path)

    dino_model = torch.hub.load(
        repo_or_dir=dino_dir,
        model="dinov3_vits16plus",
        source="local",
        weights=weights_full_path,
    )

    log.info("Building Unified DINOv3 Feature Extractor...")

    dinov3_model = build_dinov3(dino_model, config.skills.dinov3)
    dinov3_model.to(device)

    # Load Custom Weights
    head_weights_dir = config.dataset.head_weights_dir
    if head_weights_dir and os.path.exists(head_weights_dir):
        log.info(f"Loading custom head weights from {head_weights_dir}...")
        try:
            dinov3_model.depth_head.load_state_dict(
                torch.load(
                    os.path.join(head_weights_dir, "depth_head.pth"),
                    map_location=device,
                )
            )
            dinov3_model.seg_head.load_state_dict(
                torch.load(
                    os.path.join(head_weights_dir, "instance_seg_head.pt"),
                    map_location=device,
                )
            )
        except Exception as e:
            log.warning(
                f"Failed to load custom head weights from {head_weights_dir}: {e}"
            )
    else:
        log.warning(
            f"Head weights directory '{head_weights_dir}' not found. Using randomly initialized heads."
        )

    dinov3_model.eval()

    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance"
    )

    # 2. Setup Dataset
    log.info("Loading Imitation Dataset...")
    dataset = ImitationDataset(config.dataset.input_dir)
    generated_data = []

    no_mask_count = 0
    # 3. Process each sample
    for idx in tqdm(range(len(dataset)), desc="Processing Dataset"):
        img_path, wrist_pose, target_pose = dataset[idx]

        image = cv2.imread(img_path)
        if image is None:
            log.warning(f"Could not read image: {img_path}")
            continue

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure image logic matches the DINOv3 training pipeline preprocessing
        image_size = getattr(config.skills.dinov3, "image_size", 640)
        rgb_frame = cv2.resize(rgb_frame, (image_size, image_size))

        mean_np = np.array(IMG_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std_np = np.array(IMG_STD, dtype=np.float32).reshape(3, 1, 1)

        img_tensor = image_to_tensor(rgb_frame, mean_np, std_np)

        # Cast input tensor to half precision if model is fp16
        dtype = (
            torch.float16
            if getattr(config.skills.dinov3, "fp16", False)
            else torch.float32
        )
        img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=dtype)

        with torch.inference_mode():
            if config.dataset.feature_mode == "query":
                # 1. Get raw model outputs (depth and the full seg_out dict)
                _, seg_out = dinov3_model(img_tensor)

                # 2. Extract the latent queries from the decoder
                # 'transformer_decoder_last_hidden_state' has shape [Batch, Num_Queries, Query_Dim]
                all_queries = seg_out["transformer_decoder_last_hidden_state"]

                # 3. Find the 'winning' query for your target object
                class_logits = seg_out["class_queries_logits"]
                probs = torch.softmax(class_logits, dim=-1)

                # Target class ID is 0 for single-class detection
                target_class_id = 0
                query_indices = probs[..., target_class_id].argmax(
                    dim=-1
                )  # Shape: [Batch]

                # 4. Extract only that specific query vector
                batch_idx = torch.arange(all_queries.size(0))
                visual_feature_tensor = all_queries[batch_idx, query_indices]
                visual_feature = visual_feature_tensor.cpu().squeeze().numpy()

            elif config.dataset.feature_mode == "mask":
                # Extract full segmentation mask
                _, seg_out = dinov3_model(img_tensor)

                probs = torch.softmax(seg_out["class_queries_logits"], dim=-1)
                max_conf = probs[0, :, 0].max().item()

                results = processor.post_process_instance_segmentation(
                    ModelOutput(seg_out),
                    target_sizes=[(image.shape[0], image.shape[1])],
                    threshold=0.1,  # Lower threshold to see what's actually predicted
                )[0]

                # Find the target object (class 0)
                target_mask = None
                detected_classes = set()
                for seg_info in results["segments_info"]:
                    detected_classes.add(seg_info["label_id"])
                    if seg_info["label_id"] == 0:
                        target_id = seg_info["id"]
                        target_mask = (
                            (results["segmentation"] == target_id)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                        break

                if target_mask is None:
                    no_mask_count += 1
                    if idx % 50 == 0:
                        log.info(
                            f"Sample {idx}: Target class 0 not found. Max conf: {max_conf:.2f}. Detected classes: {detected_classes}"
                        )
                    continue

                visual_feature = target_mask
            else:
                raise ValueError(f"Unknown feature mode: {config.dataset.feature_mode}")

        generated_data.append(
            {
                "image_path": img_path,
                "visual_feature": visual_feature,
                "wrist_pose": wrist_pose.numpy(),
                "target_pose": target_pose.numpy(),
            }
        )

    # 4. Save Output
    log.info(
        f"Processing complete. Valid samples: {len(generated_data)} | Samples with no masks detected: {no_mask_count}"
    )

    output_file = Path(config.dataset.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving {len(generated_data)} samples to {output_file}")

    if output_file.suffix == ".pkl":
        with open(output_file, "wb") as f:
            pickle.dump(generated_data, f)
    elif output_file.suffix == ".pt":
        torch.save(generated_data, output_file)
    else:
        log.warning(f"Unknown extension {output_file.suffix}, saving as pickle.")
        with open(output_file.with_suffix(".pkl"), "wb") as f:
            pickle.dump(generated_data, f)

    log.info("Dataset generation complete!")
