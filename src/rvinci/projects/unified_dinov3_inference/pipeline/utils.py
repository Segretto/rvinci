import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def image_to_tensor(img, mean, std):
    """
    Converts an img to a tensor ready to be used in NN
    """
    img = img.astype(np.float32) / 255.
    img = (img.transpose(2,0,1) - mean) / std
    return torch.from_numpy(img)

def depth_to_colormap(
    depth_m: np.ndarray,
    dmin: float | None = None,
    dmax: float | None = None,
    colormap: int = cv2.COLORMAP_INFERNO,
    invert: bool = True,
    bgr: bool = True
) -> np.ndarray:
    """
    Convert an HxW depth map in meters to a color (BGR) visualization.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    assert depth.ndim == 2, "depth_m must be HxW"

    valid = np.isfinite(depth) & (depth > 0)

    if dmin is None or dmax is None:
        if np.any(valid):
            vals = depth[valid]
            if dmin is None:
                dmin = float(np.percentile(vals, 5))
            if dmax is None:
                dmax = float(np.percentile(vals, 95))
        else:
            dmin, dmax = 0.1, 1.0
    if dmax <= dmin:
        dmax = dmin + 1e-6

    norm = (depth - dmin) / (dmax - dmin)
    norm = np.clip(norm, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    gray8 = (norm * 255.0).astype(np.uint8)

    color_bgr = cv2.applyColorMap(gray8, colormap)

    if not np.all(valid):
        color_bgr[~valid] = (0, 0, 0)

    if bgr:
        return color_bgr
    else:
        return color_bgr[:, :, ::-1].copy()


def generate_instance_overlay(image, segmentation, segments_info,
                              class_names=None, alpha=0.6, seed=42,
                              target_size=(640, 640)):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    rng = np.random.RandomState(seed)
    palette = rng.randint(0, 256, size=(len(segments_info) + 1, 3), dtype=np.uint8)

    def overlay(base, mask, color):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        out = base.astype(np.float32)
        m3 = mask[..., None].astype(np.float32)
        color = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        out = out * (1 - alpha * m3) + color * (alpha * m3)
        return out.clip(0, 255).astype(np.uint8)

    for i, seg in enumerate(segments_info):
        inst_id = seg["id"]
        cls_id = seg["label_id"]
        score = seg.get("score", None)

        mask = segmentation == inst_id
        color = palette[i + 1]
        img = overlay(img, mask, color)

        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        ys, xs = np.nonzero(mask_np)
        if ys.size > 0:
            y0, x0 = int(ys.mean()), int(xs.mean())
            label = class_names[cls_id] if class_names else str(cls_id)
            if score is not None:
                label += f" {score:.2f}"
            cv2.putText(img, label, (x0, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    return img
