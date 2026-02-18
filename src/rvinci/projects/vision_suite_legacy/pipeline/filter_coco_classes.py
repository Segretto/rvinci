#!/usr/bin/env python3
import json
import argparse
import sys
import os

def filter_coco(input_path, output_path, include_classes=None, exclude_classes=None):
    if not os.path.exists(input_path):
        print(f"Error: Path does not exist: {input_path}")
        return

    with open(input_path, 'r') as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    
    # Identify target category IDs
    if include_classes:
        target_cats = [cat for cat in categories if cat["name"] in include_classes]
    elif exclude_classes:
        target_cats = [cat for cat in categories if cat["name"] not in exclude_classes]
    else:
        target_cats = categories

    target_cat_ids = {cat["id"] for cat in target_cats}
    
    # Filter annotations
    annotations = coco.get("annotations", [])
    filtered_annotations = [ann for ann in annotations if ann["category_id"] in target_cat_ids]
    
    # Construct new coco object
    new_coco = coco.copy()
    new_coco["categories"] = target_cats
    new_coco["annotations"] = filtered_annotations
    
    # We keep images as they are, as they might be needed for consistency 
    # even if they have no annotations for the current filtered classes.
    
    with open(output_path, 'w') as f:
        json.dump(new_coco, f, indent=4)
        
    print(f"Filtered COCO saved to: {output_path}")
    print(f"Remaining categories: {[cat['name'] for cat in target_cats]}")
    print(f"Remaining annotations: {len(filtered_annotations)} (from {len(annotations)})")

def main():
    parser = argparse.ArgumentParser(description="Filter COCO classes.")
    parser.add_argument("input_json", help="Path to source COCO JSON.")
    parser.add_argument("output_json", help="Path to save filtered COCO JSON.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include", nargs="+", help="List of class names to keep.")
    group.add_argument("--exclude", nargs="+", help="List of class names to remove.")
    
    args = parser.parse_args()
    
    if not args.include and not args.exclude:
        print("Warning: No include or exclude classes specified. Copying file as is.")
        
    filter_coco(args.input_json, args.output_json, include_classes=args.include, exclude_classes=args.exclude)

if __name__ == "__main__":
    main()
