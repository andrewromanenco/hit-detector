import cv2
import argparse
import os
import csv

def main():
    parser = argparse.ArgumentParser(description="Mark points on an image and log their coordinates.")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("csv_output", help="Path to output CSV file for coordinates")
    parser.add_argument("--radius", type=int, default=15, help="Radius of the marker circle (default: 15)")
    parser.add_argument("--color", choices=["red", "green"], default="red", help="Circle color (default: red)")
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"❌ Could not load image: {args.image_path}")
        exit(1)

    base, ext = os.path.splitext(args.image_path)
    output_image_path = f"{base}_marked{ext}"

    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0)
    }
    circle_color = color_map[args.color]

    display_img = img.copy()

    csv_file = open(args.csv_output, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["x", "y"])  # Header if it's a new file

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(display_img, (x, y), args.radius, circle_color, 2)
            cv2.imshow("Click to mark", display_img)
            csv_writer.writerow([x, y])
            print(f"🖊️  Marked at ({x}, {y})")

    # === Launch interactive window ===
    cv2.imshow("Click to mark", display_img)
    cv2.setMouseCallback("Click to mark", on_mouse)

    print("🖱️ Click on the image to mark. Press any key to finish.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    csv_file.close()

    cv2.imwrite(output_image_path, display_img)
    print(f"✅ Marked image saved to: {output_image_path}")
    print(f"📄 Coordinates saved to: {args.csv_output}")
