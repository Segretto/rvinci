#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont


from rvinci.libs.visualization.drawing import draw_legend, draw_horizontal_legend, get_font
from rvinci.libs.visualization.palette import PaletteManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_rounded_corners(im, rad):
    if rad <= 0: return im
    mask = Image.new('L', im.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, im.size[0], im.size[1]), radius=rad, fill=255)
    im.putalpha(mask)
    return im

def main():
    parser = argparse.ArgumentParser(description="Generate a grid of worst-case error images.")
    parser.add_argument("input_dir", help="Directory containing images following <class>_worst_<type>.png")
    parser.add_argument("--output", default="worst_cases_grid.png", help="Output filename.")
    parser.add_argument("--padding", type=int, default=10, help="Outer padding in pixels.")
    parser.add_argument("--radius", type=int, default=20, help="Corner radius for images.")
    parser.add_argument("--whitespace", type=int, default=10, help="Space between images.")
    parser.add_argument("--legend_style", choices=["standard", "alternate", "none"], default="standard", 
                        help="Legend style: 'standard' (top-left vertical), 'alternate' (bottom-centered horizontal), or 'none'.")
    parser.add_argument("--font_size", type=int, default=24, help="Font size for the grid legend.")
    parser.add_argument("--palette_config", help="Optional path to a palette config file (class - type: #hex).")
    parser.add_argument("--classes", nargs="+", help="Optional space-separated list of classes to include and their order.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Scan for images: <class>_worst_<type>.png
    # Pattern: ^(.+)_(fp|fn)\.png$
    pattern = re.compile(r"^(.+)_(fp|fn)\.png$")
    
    grid_images = defaultdict(dict)
    all_classes = set()
    
    for fname in os.listdir(args.input_dir):
        match = pattern.match(fname)
        if match:
            cls_name = match.group(1)
            err_type = match.group(2)
            img_path = os.path.join(args.input_dir, fname)
            grid_images[cls_name][err_type] = Image.open(img_path)
            all_classes.add(cls_name)
    
    if not grid_images:
        logger.warning("No matching images found in the input directory.")
        return

    # Determine classes to show
    if args.classes:
        sorted_classes = [c for c in args.classes if c in all_classes]
        if not sorted_classes:
            logger.error(f"None of the provided classes {args.classes} found in {args.input_dir}")
            sys.exit(1)
    else:
        # Default alphabetical
        sorted_classes = sorted(list(all_classes))

    # Initialize PaletteManager
    palette_manager = PaletteManager(args.palette_config)
    
    # --- Image Normalization (Orientation & Size) ---
    all_loaded_imgs_pre = []
    for cls_dict in grid_images.values():
        all_loaded_imgs_pre.extend(list(cls_dict.values()))
        
    if not all_loaded_imgs_pre:
        return

    # 1. Determine target orientation (Landscape vs Portrait) by majority vote
    num_landscape = sum(1 for im in all_loaded_imgs_pre if im.size[0] >= im.size[1])
    target_is_landscape = num_landscape >= (len(all_loaded_imgs_pre) / 2)
    
    # 2. Normalize orientations and collect normalized sizes
    norm_widths = []
    norm_heights = []
    for cls_name in grid_images:
        for err_type in grid_images[cls_name]:
            im = grid_images[cls_name][err_type]
            im_is_landscape = im.size[0] >= im.size[1]
            if im_is_landscape != target_is_landscape:
                im = im.rotate(90, expand=True)
                grid_images[cls_name][err_type] = im
            norm_widths.append(im.size[0])
            norm_heights.append(im.size[1])

    # 3. Determine median size from normalized images
    cell_w = sorted(norm_widths)[len(norm_widths)//2]
    cell_h = sorted(norm_heights)[len(norm_heights)//2]
    
    logger.info(f"Normalizing to {'Landscape' if target_is_landscape else 'Portrait'} ({cell_w}x{cell_h})")

    # 4. Final resize to ensure exact matching resolution
    for cls_name in grid_images:
        for err_type in grid_images[cls_name]:
            im = grid_images[cls_name][err_type]
            if im.size != (cell_w, cell_h):
                grid_images[cls_name][err_type] = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)

    types = ["fp", "fn"]
    
    # Calculate grid dimensions 
    grid_w = args.padding * 2 + args.whitespace + cell_w * 2
    grid_h = args.padding * 2 + args.whitespace * (len(sorted_classes) - 1) + cell_h * len(sorted_classes)
    
    # Reserve space for legend if alternate
    legend_reserve_h = 0
    if args.legend_style == "alternate":
        legend_reserve_h = 100 # Estimated height for horizontal legend area
        grid_h += legend_reserve_h

    summary_grid = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))
    draw_grid = ImageDraw.Draw(summary_grid)
    
    # Load font
    grid_font = get_font(size=args.font_size, bold=True)

    logger.info(f"Assembling grid for classes: {sorted_classes}")

    for row_idx, cls_name in enumerate(sorted_classes):
        row_y = args.padding + row_idx * (cell_h + args.whitespace)
        
        # 1. Add images
        for col_idx, t in enumerate(types):
            if t in grid_images[cls_name]:
                img_cell = grid_images[cls_name][t].convert("RGBA")
                img_cell = apply_rounded_corners(img_cell, args.radius)
                
                x_pos = args.padding + col_idx * (cell_w + args.whitespace)
                summary_grid.paste(img_cell, (x_pos, row_y), img_cell)
            else:
                logger.debug(f"Missing image for {cls_name} {t}")

    # 2. Draw unified legend for ALL classes
    full_class_map = {}
    idx = 0
    error_types = ["Correct", "False Positive", "False Negative"]
    for cls_name in sorted_classes:
        for et in error_types:
            color = palette_manager.get_color(cls_name, et)
            full_class_map[idx] = {
                "name": f"{cls_name.capitalize()} - {et}",
                "color": color
            }
            idx += 1
    
    if args.legend_style == "standard":
        # Legend at absolute top-left of the first image
        legend_margin = max(5, args.radius // 2)
        draw_legend(draw_grid, full_class_map, grid_font, grid_w, grid_h, radius=10, 
                    x=args.padding + legend_margin, y=args.padding + legend_margin)
    elif args.legend_style == "alternate":
        # Legend at bottom centered
        legend_y = grid_h - legend_reserve_h + 20
        draw_horizontal_legend(draw_grid, full_class_map, grid_font, grid_w, grid_h, y=legend_y)
    # Style "none" does nothing

    # Save final grid
    if summary_grid.mode == 'RGBA':
        summary_grid = summary_grid.convert('RGB')
    
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    summary_grid.save(args.output)
    logger.info(f"Saved summary grid to {args.output}")

if __name__ == "__main__":
    main()
