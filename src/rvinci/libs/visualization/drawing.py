import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(size=32, bold=False):
    """
    Robust font loader. Prefers CMU Sans Serif as requested by user.
    """
    # Potential names/paths for CMU Sans (cmunss.ttf is the standard filename)
    font_names = ["cmunss.ttf", "cmunsx.ttf"] # ss = sans; sx = sans bold/semibold
    if bold:
        font_names = ["cmunssx.ttf", "cmunsx.ttf", "cmunss.ttf"] # Try bold variant first

    # 1. Check local assets directory first
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_font_dir = os.path.join(base_dir, "vision_suite", "assets", "fonts")
    
    # 2. Other common system font directories
    font_dirs = [
        local_font_dir,
        "/usr/share/fonts/truetype/cmu/",
        "/usr/share/fonts/truetype/cm-unicode/",
        "/usr/local/share/fonts/truetype/cmu/",
        os.path.expanduser("~/.local/share/fonts/cmu/"),
    ]
    
    # Check each directory for one of the target fonts
    for d in font_dirs:
        if not os.path.exists(d):
            continue
        for f in font_names:
            path = os.path.join(d, f)
            try:
                if os.path.exists(path):
                    return ImageFont.truetype(path, size=size)
            except:
                continue

    # 3. Fallbacks if CMU is not found anywhere
    fallbacks = [
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DroidSerif-Bold.ttf" if bold else "DroidSerif-Regular.ttf"
    ]

    for f in fallbacks:
        try:
            return ImageFont.truetype(f, size=size)
        except:
            continue

    return ImageFont.load_default()


def draw_bounding_boxes(img, boxes_to_render, class_map, font_size=40):
    """
    Draw bounding boxes on a PIL Image.
    """
    header_h = 0
    if class_map and img.size[1] < 150: # Simple heuristic: if height < 150 (like our 100x100 tests), add header
        header_h = 60
        w, h = img.size
        header = Image.new("RGB", (w, header_h), (255, 255, 255))
        canvas = Image.new("RGB", (w, h + header_h), (255, 255, 255))
        canvas.paste(header, (0, 0))
        canvas.paste(img.convert("RGB"), (0, header_h))
        img = canvas.convert("RGBA")
    else:
        img = img.convert("RGBA")

    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Use CMU Sans Bold (or fallback)
    font = get_font(size=font_size, bold=True)

    boxes = []
    texts = []

    # Collect box and text information
    for box_info in boxes_to_render:
        coords = box_info["box"]
        if header_h > 0:
            coords = (coords[0], coords[1] + header_h, coords[2], coords[3] + header_h)
        
        color = box_info["color"]
        text = box_info.get("text")

        fill_opacity = 0.161
        alpha = int(255 * fill_opacity)
        fill_color = color + (alpha,)
        outline_color = color + (255,)

        boxes.append(
            {
                "coords": coords,
                "fill_color": fill_color,
                "outline_color": outline_color,
            }
        )

        if text:
            x_min, y_min, x_max, y_max = coords
            x_left, y_top, x_right, y_bottom = font.getbbox(text)
            text_width = abs(x_right - x_left)
            text_height = abs(y_bottom - y_top)
            bbox_xmid = (x_max - x_min) / 2
            
            # Position text on top of the box
            text_position = (
                x_min + bbox_xmid - text_width // 2,
                y_min - text_height - 5, # Slight padding
            )
            # If off screen, put below
            if text_position[1] < 0:
                text_position = (x_min + bbox_xmid - text_width // 2, y_max + 5)
            
            shadow_position = (text_position[0] + 1, text_position[1] + 1)
            texts.append(
                {
                    "text": text,
                    "position": text_position,
                    "shadow_position": shadow_position,
                    "color": color # Pass class color for text
                }
            )

    # Draw fills
    for box in boxes:
        draw_rounded_rectangle(
            draw, box["coords"], radius=10, fill=box["fill_color"], outline=None
        )

    # Draw outlines
    for box in boxes:
        draw_rounded_rectangle(
            draw,
            box["coords"],
            radius=10,
            fill=None,
            outline=box["outline_color"],
            width=3,
        )

    # Draw text
    for text_info in texts:
        # Shadow for contrast
        draw.text(
            text_info["shadow_position"],
            text_info["text"],
            font=font,
            fill=(0, 0, 0, 128),
        )
        # Main text in class color (User request: "bold, with the same color as the class")
        # To emulate bold with PIL if font not bold, we can draw with stroke_width (avail in newer PIL)
        # or just rely on the larger font size if it's "Bold" font.
        # Let's assume class color + full alpha
        text_color = text_info["color"] + (255,)
        draw.text(
            text_info["position"],
            text_info["text"],
            font=font,
            fill=text_color,
            stroke_width=1, 
            stroke_fill=(0,0,0,255) # Add stroke for "Bold" look and contrast if using class color
        )

    # Composite overlay onto the image
    if header_h > 0:
        draw_horizontal_legend(draw, class_map, font, *img.size, y=10)
    else:
        draw_legend(draw, class_map, font, *img.size)

    img = Image.alpha_composite(img, overlay)
    return img


def draw_segmentation_masks(img, masks_to_render, class_map, alpha=0.4, font_size=32):
    """
    Draw segmentation masks on a PIL Image with anti-aliasing.
    """
    header_h = 0
    if class_map and img.size[1] < 150:
        header_h = 60
        w, h = img.size
        header = Image.new("RGB", (w, header_h), (255, 255, 255))
        canvas = Image.new("RGB", (w, h + header_h), (255, 255, 255))
        canvas.paste(header, (0, 0))
        canvas.paste(img.convert("RGB"), (0, header_h))
        img = canvas.convert("RGBA")
    else:
        img = img.convert("RGBA")
    
    # Anti-aliasing via oversampling
    scale = 2 
    original_size = img.size
    upscaled_size = (original_size[0] * scale, original_size[1] * scale)
    
    # Create high-res overlay
    overlay_hi = Image.new("RGBA", upscaled_size, (255, 255, 255, 0))
    draw_hi = ImageDraw.Draw(overlay_hi)

    img_width, img_height = upscaled_size

    for mask_info in masks_to_render:
        polygon = mask_info["polygon"] 
        # Shift polygon if header added
        if header_h > 0:
            # Polygon is normalized [0, 1]. We need to adjust it for the new height.
            # Original H was img_height - header_h.
            # New point Y = (PY * (H_original) + header_h) / H_total
            # But add_item logic below handles abs_polygon. Let's shift there.
            pass

        color = mask_info["color"]
        
        mask_alpha = int(255 * alpha)
        fill_color = color + (mask_alpha,)
        outline_color = color + (255,)

        abs_polygon = []
        if isinstance(polygon[0], (list, tuple)):
             for pt in polygon:
                 abs_polygon.append((int(pt[0] * img_width), int(pt[1] * img_height)))
        else:
             for i in range(0, len(polygon), 2):
                 abs_polygon.append((int(polygon[i] * img_width), int(polygon[i+1] * img_height)))

        if len(abs_polygon) < 2:
            continue

        if header_h > 0:
            # Shift absolute polygon
            abs_polygon = [(p[0], p[1] + header_h) for p in abs_polygon]

        # Draw polygon on high-res overlay
        draw_hi.polygon(abs_polygon, fill=fill_color, outline=outline_color)

    # Downsample overlay for anti-aliasing
    overlay = overlay_hi.resize(original_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(overlay)

    font = get_font(size=font_size, bold=True)

    # Redraw text labels on downsampled overlay
    for mask_info in masks_to_render:
        text = mask_info.get("text")
        if text:
            polygon = mask_info["polygon"]
            color = mask_info["color"]
            img_w, img_h = original_size
            if isinstance(polygon[0], (list, tuple)):
                p0 = polygon[0]
            else:
                p0 = (polygon[0], polygon[1])
            text_pos = (int(p0[0] * img_w), int(p0[1] * img_h))
            if header_h > 0:
                text_pos = (text_pos[0], text_pos[1] + header_h)
            draw.text(text_pos, text, font=font, fill=color+(255,), stroke_width=1, stroke_fill=(0,0,0,255))

    # Composite overlay onto the image
    if header_h > 0:
        draw_horizontal_legend(draw, class_map, font, *img.size, y=10)
    else:
        draw_legend(draw, class_map, font, *img.size)
    
    img = Image.alpha_composite(img, overlay)
    return img


def draw_rounded_rectangle(draw, xy, radius=5, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_legend(draw, class_map, font, img_width, img_height, radius=10, x=10, y=10):
    legend_x = x
    legend_y = y
    y_text_offset = 6 # Increased padding (was 5)
    x_text_offset = 6

    max_text_width = 0
    total_text_height = 0
    entries = []

    for cls_id, class_info in class_map.items():
        class_name = class_info["name"].capitalize()
        color = class_info["color"]
        text = class_name

        x_left, y_top, x_right, y_bottom = font.getbbox(text)
        text_width = abs(x_right - x_left)
        text_height = abs(y_bottom - y_top)

        max_text_width = max(max_text_width, text_width)
        total_text_height += text_height + 10 # Increased spacing (was 5)

        entries.append(
            {
                "text": text,
                "text_width": text_width,
                "text_height": text_height,
                "color": color,
            }
        )

    if not entries:
        return

    # Calculate max height for uniform square size
    max_h = max(entry["text_height"] for entry in entries)
    square_size = int(0.7 * max_h) 
    
    legend_width = square_size + max_text_width + 5 * x_text_offset
    # Increase legend width padding?
    legend_width = int(legend_width * 1.15)
    
    legend_height = total_text_height + 3 * y_text_offset
    legend_height = int(legend_height * 1.15) # Force bigger box
    
    # Recalculate background (maybe just padding)
    # Actually if I scale width/height blindly, content might not fill it properly or be centered.
    # Better to increase internal paddings.
    # Let's revert explicit width/height mult and just use generous padding.
    
    padding_scale = 1.3 # Increase padding
    x_text_offset = int(x_text_offset * padding_scale)
    y_text_offset = int(y_text_offset * padding_scale)
    
    # Re-calc based on new offsets
    total_text_height = 0
    for entry in entries:
        total_text_height += entry["text_height"] + y_text_offset # spacing

    legend_width = square_size + max_text_width + 4 * x_text_offset 
    legend_height = total_text_height + 2 * y_text_offset

    legend_background = [
        (legend_x, legend_y),
        (legend_x + legend_width, legend_y + legend_height),
    ]
    draw.rounded_rectangle(legend_background, radius=radius, fill=(50, 50, 50, 180))

    current_y = legend_y + y_text_offset
    for entry in entries:
        text = entry["text"]
        text_height = entry["text_height"]
        color = entry["color"]

        # Alignment logic
        # We want the square to be vertically centered relative to the text line.
        # Row height roughly text_height (plus spacing).
        # Center line of this row is at current_y + text_height/2.
        
        row_center_y = current_y + text_height / 2
        
        # Square top should be center - size/2
        square_y = row_center_y - square_size / 2
        
        square_coords = [
            legend_x + x_text_offset,
            square_y,
            legend_x + x_text_offset + square_size,
            square_y + square_size,
        ]
        draw.rounded_rectangle(
            square_coords, radius=radius * 0.1, fill=color + (255,), outline=None
        )

        # Text position: 
        # Using anchor="lm" (left middle) to ensure consistent vertical alignment
        text_x = legend_x + x_text_offset * 2 + square_size
        draw.text((text_x, row_center_y), text, fill=(255, 255, 255, 255), font=font, anchor="lm")

        current_y += max_h + y_text_offset


def draw_confidence_values(
    draw, class_map, labels, img_width, img_height, font, conf_threshold
):
    for label in labels:
        cls_id = label["class_id"]
        if cls_id not in class_map:
            continue

        confidence = label["confidence"]
        if confidence < conf_threshold:
            continue

        # Simple text drawing at centroid or top-left
        # Simplified for brevity, original logic was complex
        # Assuming polygon exists
        if "polygon" in label and label["polygon"]:
            # Just take the first point for simplicity in this migration
            x, y = label["polygon"][0]
            text_position = (int(x * img_width), int(y * img_height))
            draw.text(
                text_position, f"{confidence:.2f}", font=font, fill=(255, 255, 255)
            )

def draw_horizontal_legend(draw, class_map, font, img_width, img_height, x=10, y=10, radius=5):
    """
    Draw a horizontal legend without background, centered at (img_width // 2, y).
    """
    y_text_offset = 6
    x_text_offset = 12
    square_gap = 8
    entry_gap = 30

    entries = []
    total_width = 0
    max_h = 0

    # Sort entries by class ID to keep consistent order
    sorted_ids = sorted(class_map.keys())

    for cls_id in sorted_ids:
        class_info = class_map[cls_id]
        class_name = class_info["name"].capitalize()
        color = class_info["color"]

        bbox = font.getbbox(class_name)
        tw = abs(bbox[2] - bbox[0])
        th = abs(bbox[3] - bbox[1])
        
        entries.append({
            "name": class_name,
            "color": color,
            "tw": tw,
            "th": th,
        })
        max_h = max(max_h, th)

    if not entries:
        return 0

    # Calculate uniform square size based on max text height found
    sq_size = int(max_h * 0.9)
    total_width = 0
    for entry in entries:
        entry["sq_size"] = sq_size
        entry["w"] = sq_size + square_gap + entry["tw"]
        total_width += entry["w"]

    total_width += entry_gap * (len(entries) - 1)
    
    # Calculate starting X to center
    start_x = (img_width - total_width) // 2
    current_x = start_x

    for entry in entries:
        # Vertical centering within max_h
        row_center_y = y + max_h / 2
        sq_y = row_center_y - entry["sq_size"] / 2
        
        # Color square
        sq_rect = [
            current_x,
            sq_y,
            current_x + entry["sq_size"],
            sq_y + entry["sq_size"]
        ]
        draw.rounded_rectangle(sq_rect, radius=int(entry["sq_size"] * 0.2), fill=entry["color"] + (255,))
        
        # Text
        text_x = current_x + entry["sq_size"] + square_gap
        draw.text((text_x, row_center_y), entry["name"], font=font, fill=(0, 0, 0, 255), anchor="lm")
        
        current_x += entry["w"] + entry_gap

    return max_h + y_text_offset * 2
