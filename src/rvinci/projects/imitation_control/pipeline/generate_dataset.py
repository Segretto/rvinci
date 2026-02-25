import os
import cv2
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor

from rvinci.core.logging import get_logger
from rvinci.skills.perception.dinov3.api import build_dinov3
from rvinci.skills.manipulation.imitation.api import ImitationDataset

log = get_logger(__name__)


def generate_pipeline(config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # 1. Setup Model
    log.info("Loading DINOv3 Backbone...")
    # Using the standard Vits14 as default for this project
    dino_model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model="dinov2_vits14",
    )

    log.info("Building Unified DINOv3 Feature Extractor...")
    dinov3_model = build_dinov3(dino_model, config.skills.dinov3)
    dinov3_model.to(device)
    dinov3_model.eval()

    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance"
    )

    # 2. Setup Dataset
    input_dir = config.dataset.input_dir
    log.info(f"Loading Imitation Dataset from {input_dir}")
    try:
        dataset = ImitationDataset(input_dir)
    except FileNotFoundError as e:
        log.error(e)
        return

    generated_data = []

    # 3. Process Images
    for idx in tqdm(range(len(dataset)), desc="Processing Dataset"):
        img_path, wrist_pose, target_pose = dataset[idx]

        if not os.path.exists(img_path):
            log.warning(f"Image not found: {img_path}")
            continue

        # Load and preprocess image
        image = cv2.imread(img_path)
        if image is None:
            log.warning(f"Failed to read image: {img_path}")
            continue

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure image logic matches the DINOv3 pipeline which uses 518x518 standard
        image_size = getattr(config.skills.dinov3, "image_size", 518)
        rgb_frame = cv2.resize(rgb_frame, (image_size, image_size))

        # Use processor for standard Mask2Former normalization if doing masks
        # Force don't resize so DINOv2 patch embedding (14x14) works with our 518x518 standard
        inputs = processor(
            images=rgb_frame,
            return_tensors="pt",
            do_resize=False,
        ).to(device)

        with torch.inference_mode():
            if config.dataset.feature_mode == "query":
                # 1. Get raw model outputs (depth and the full seg_out dict)
                _, seg_out = dinov3_model(inputs.pixel_values)

                # 2. Extract the latent queries from the decoder
                # 'transformer_decoder_last_hidden_state' has shape [Batch, Num_Queries, Query_Dim]
                all_queries = seg_out["transformer_decoder_last_hidden_state"]

                # 3. Find the 'winning' query for your target object
                # seg_out.class_queries_logits has shape [Batch, Num_Queries, Num_Classes + 1]
                class_logits = seg_out["class_queries_logits"]
                probs = torch.softmax(class_logits, dim=-1)

                # Let's say your target class ID is 1 (e.g., "handle" or "object")
                target_class_id = 1
                query_indices = probs[..., target_class_id].argmax(
                    dim=-1
                )  # Shape: [Batch]

                # 4. Extract only that specific query vector
                batch_idx = torch.arange(all_queries.size(0))
                visual_feature_tensor = all_queries[batch_idx, query_indices]
                visual_feature = visual_feature_tensor.cpu().squeeze().numpy()

            elif config.dataset.feature_mode == "mask":
                # Extract full segmentation mask
                _, seg_logits = dinov3_model(inputs.pixel_values)
                results = processor.post_process_instance_segmentation(
                    seg_logits,
                    target_sizes=[(image.shape[0], image.shape[1])],
                )[0]
                visual_feature = results["segmentation"].cpu().numpy()
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
