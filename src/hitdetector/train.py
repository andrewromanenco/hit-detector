import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from hitdetector.dataset import TargetDataset

from sklearn.metrics import precision_recall_fscore_support


class BinaryTargetDataset(TargetDataset):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        binary_label = torch.tensor(0 if label.item() in [0] else 1, dtype=torch.float32)
        return image, binary_label


class SimpleCNN(nn.Module):
    def __init__(self, sample_input):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy_output = self.features(sample_input.unsqueeze(0))
            self.flattened_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            #nn.Sigmoid() # lesson learned, don't forget to remove before balancing
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train(hits_dir, blanks_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BinaryTargetDataset(hits_dir, blanks_dir)

    sample_input, _ = dataset[0]
    model = SimpleCNN(sample_input).to(device)

    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    num_zeros = (targets == 0).sum()
    num_ones = (targets == 1).sum()
    pos_weight = num_ones / num_zeros
    print(f"Class balance: 0 -> {num_zeros.item()}, 1 -> {num_ones.item()}, pos_weight = {pos_weight:.2f}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 1000
    patience = 5
    best_accuracy = 0.0
    epochs_no_improve = 0
    grace_epochs = 10

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        model.train()
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()

            all_preds.append(predicted.cpu())
            all_labels.append(y.cpu())

            correct += (predicted == y).sum().item()
            total += y.size(0)

        preds_tensor = torch.cat(all_preds).squeeze().numpy()
        labels_tensor = torch.cat(all_labels).squeeze().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_tensor, preds_tensor, average="binary", zero_division=0
        )

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {total_loss / total:.4f} - "
              f"Accuracy: {correct / total:.4f} - "
              f"Precision: {precision:.4f} - "
              f"Recall: {recall:.4f} - "
              f"F1-score: {f1:.4f}")

        accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            model_path = Path(model_path)
            model.patch_size = sample_input.shape[-1]
            torch.save(model, model_path)
        elif epoch <= grace_epochs:
            pass
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}. Best accuracy: {best_accuracy:.4f}")
                break
    print(f"✅ Model saved to {model_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train binary CNN target detector")
    parser.add_argument("--hits-dir", required=True, help="Path to directory with hit patches")
    parser.add_argument("--blanks-dir", required=True, help="Path to directory with blank patches")
    parser.add_argument("--model-path", required=True, help="Path to save the trained model")

    args = parser.parse_args()
    train(args.hits_dir, args.blanks_dir, args.model_path)

if __name__ == "__main__":
    main()
