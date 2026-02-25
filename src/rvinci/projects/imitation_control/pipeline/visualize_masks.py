import cv2

from rvinci.projects.imitation_control.visualize_masks import VisProjectConfig
from rvinci.skills.manipulation.imitation.datasets.dataset import OfflineFeatureDataset
from rvinci.libs.visualization.drawing import generate_instance_overlay
from rvinci.core.logging import get_logger

log = get_logger(__name__)


def visualize_pipeline(config: VisProjectConfig) -> None:
    dataset = OfflineFeatureDataset(config.visualize.data_path)

    log.info(f"Loaded {len(dataset)} items from {config.visualize.data_path}.")
    log.info("Press 'x' to skip, 'q' or 'ESC' to quit, any other key to advance.")

    try:
        for i in range(len(dataset)):
            visual_feature, _, _, img_path = dataset[i]

            # We expect mask mode here, so visual_feature should be a 2D map HxW
            # OfflineDataset loaded it as a Float tensor...
            mask = visual_feature.numpy().astype(int)

            if len(mask.shape) != 2:
                log.error(
                    f"visual_feature has shape {mask.shape}, expected 2D mask. Was dataset generated with feature_mode='mask'?"
                )
                return

            if not img_path:
                log.warning(f"No original image path found for sample {i}.")
                continue

            # 1. Load Original RGB Image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                log.warning(f"Failed to load image at {img_path}")
                continue

            # 2. Re-create the overlay using agnostic tools
            # `generate_instance_overlay` uses segments_info: list of dicts with 'id', 'label_id'
            # our mask currently has binary 1 or 0 since we isolated the target object.
            segments_info = [{"id": 1, "label_id": 1, "score": 1.0}]

            # Map class ID 1 -> "Target"
            class_names = {1: "Target"}

            overlay_img = generate_instance_overlay(
                image=img_bgr,
                segmentation=mask,
                segments_info=segments_info,
                class_names=class_names,
                alpha=0.6,
                target_size=(img_bgr.shape[1], img_bgr.shape[0]),
            )

            cv2.imshow(
                "Mask Visualization", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            )

            key = cv2.waitKey(0) & 0xFF
            if key in [27, ord("q")]:
                log.info("Visualization terminated by user.")
                break
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received. Gracefully exiting visualization.")
    finally:
        cv2.destroyAllWindows()
        log.info("Visualization finished.")
