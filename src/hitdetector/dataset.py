from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class TargetDataset(Dataset):
    def __init__(self, hits_dir, blanks_dir):
        self.class_map = {
            "blank": 0,
            "hit": 1
        }

        class_to_folder = {
            "hit": hits_dir,
            "blank": blanks_dir
        }

        self.transform = T.Compose([
            T.Grayscale(),
            T.ToTensor()
        ])

        self.samples = []

        self.flip_types = ['none', 'h', 'v']  # none, horizontal, vertical
        self.angles = [0, 90, 180, 270]

        for class_name, label in self.class_map.items():
            folder = Path(class_to_folder[class_name])
            for img_path in folder.glob("*.png"):
                for flip in self.flip_types:
                    for angle in self.angles:
                        self.samples.append((img_path, label, flip, angle))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, flip, angle = self.samples[idx]

        image = Image.open(path).convert("L")  # grayscale
        if flip == 'h':
            image = TF.hflip(image)
        elif flip == 'v':
            image = TF.vflip(image)

        image = self.transform(image)  # convert to tensor after all pre processing is done
        if angle != 0:
            image = TF.rotate(image, angle)

        return image, torch.tensor(label, dtype=torch.long)
