import os
import sys
import csv
import cv2
import uuid
import argparse
import numpy as np


def extract_patches(image_path, csv_path, patch_size, output_folder):
    if not os.path.isfile(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    if not os.path.isfile(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        sys.exit(1)

    height, width = image.shape[:2]
    half = patch_size // 2
    total_saved = 0

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            try:
                x = int(float(row['x']))
                y = int(float(row['y']))
            except KeyError:
                print("❌ CSV must contain headers 'x' and 'y'")
                sys.exit(1)
            except ValueError:
                print(f"⚠️ Skipping invalid coordinate at row {idx + 2}")
                continue

            x1, y1 = x - half, y - half
            x2, y2 = x + half, y + half

            if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
                print(f"⚠️ Skipping patch at ({x}, {y}) - out of bounds")
                continue

            patch = image[y1:y2, x1:x2]
            filename = os.path.join(output_folder, f"{uuid.uuid4().hex}.png")
            cv2.imwrite(filename, patch)
            total_saved += 1

    print(f"✅ Saved {total_saved} patches to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Extract patches from image based on coordinates CSV")
    parser.add_argument("image", help="Input image file")
    parser.add_argument("csv", help="CSV file with coordinates (must have headers 'x' and 'y')")
    parser.add_argument("size", type=int, help="Patch size (square)")
    parser.add_argument("output", help="Output folder (created if it does not exist)")
    args = parser.parse_args()

    extract_patches(args.image, args.csv, args.size, args.output)


if __name__ == "__main__":
    main()
