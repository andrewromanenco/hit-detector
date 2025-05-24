import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Run sliding window prediction on an image")
    parser.add_argument("model_path", type=str, help="Path to the trained model file (.pt or .pth)")
    parser.add_argument("input_image", type=str, help="Input image file")
    parser.add_argument("output_image", type=str, help="Output image file (must not already exist)")
    parser.add_argument("--color", type=str, default="FF0000", help="Color for marking matches (6-char hex RGB)")
    parser.add_argument("--opacity", type=int, default=128, help="Opacity of overlay (0-255)")
    parser.add_argument("--target_label", type=int, choices=[0, 1], default=1, help="Label to highlight (0 or 1)")
    return parser.parse_args()


def hex_to_rgb(hex_color):
    hex_color = hex_color.strip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    if not hasattr(model, "patch_size"):
        raise AttributeError("Loaded model does not have a 'patch_size' attribute")
    return model, model.patch_size


def run_inference(
    model: torch.nn.Module,
    image: Image.Image,
    original: Image.Image,
    color: tuple,
    opacity: int,
    target_label: int,
    patch_size: int,
    stride: int = 4
):
    transform = transforms.ToTensor()

    width, height = image.size
    total_patches = ((width - patch_size) // stride + 1) * ((height - patch_size) // stride + 1)
    done = 0
    last_percent_reported = -1

    # Prepare overlay image with transparency (same size as original, RGBA)
    overlay = Image.new("RGBA", original.size, (0, 0, 0, 0))

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            tensor = transform(patch).unsqueeze(0)  # shape: [1, 1, patch_size, patch_size]
            with torch.no_grad():
                pred = model(tensor)
                predicted_label = int(pred.item() > 0.9)

            if predicted_label == target_label:
                patch_overlay = Image.new("RGBA", (patch_size, patch_size), color + (opacity,))
                overlay.paste(patch_overlay, (x, y), patch_overlay)

            done += 1
            percent = int(done / total_patches * 100)
            if percent != last_percent_reported:
                print(f"\rProgress: {percent:3d}% ", end="", flush=True)
                last_percent_reported = percent

    print("\nDone.")

    blended = Image.alpha_composite(original.convert("RGBA"), overlay)
    return blended.convert("RGB")


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    input_path = Path(args.input_image)
    output_path = Path(args.output_image)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output image already exists: {output_path}")

    color = hex_to_rgb(args.color)
    grayscale_img = Image.open(input_path).convert("L")
    original_img = Image.open(input_path).convert("RGB")

    model, patch_size = load_model(model_path)
    marked_img = run_inference(model, grayscale_img, original_img, color, args.opacity, args.target_label, patch_size)

    marked_img.save(output_path)
    print(f"Saved marked image to {output_path.resolve()}")


if __name__ == "__main__":
    main()
