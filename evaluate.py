import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import CinemaClassifier

# Configurations
DATA_DIR = "data"
MODEL_PATH = "best_model.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    class_names = checkpoint["classes"]
    model = CinemaClassifier(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.2%})")
    return model, class_names

def get_val_loader():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=DATA_DIR, transform=transform)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    _, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    return DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False), dataset.classes


def evaluate(model, loader, class_names):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix - Cinema Look Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
    print("Saved: confusion_matrix.png")

    return all_preds, all_labels

def show_sample_predictions(model, loader, class_names, n=8):
    images_shown, preds_shown, labels_shown = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            confs, predicted = torch.max(probs, 1)

            for i in range(len(images)):
                images_shown.append(images[i])
                preds_shown.append((predicted[i].item(), confs[i].item()))
                labels_shown.append(labels[i].item())
                if len(images_shown) >= n:
                    break
            if len(images_shown) >= n:
                break


    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cols = 4
    rows = (n + cols -1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(n):
        img = images_shown[i] * std + mean # Denormalize
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()

        pred_idx, conf = preds_shown[i]
        true_idx = labels_shown[i]
        pred_name = class_names[pred_idx]
        true_name = class_names[true_idx]
        correct = pred_idx == true_idx

        axes[i].imshow(img)
        axes[i].set_title(
            f"Pred: {pred_name} ({conf:.0%})\nTrue: {true_name}",
            color="green" if correct else "red",
            fontsize=9
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions (green = correct, red = incorrect)", fontsize=11)
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=300)
    plt.show()
    print("Saved: sample_predictions.png")


def main():
    print(f"Using device: {DEVICE}\n")

    model, class_names = load_model(MODEL_PATH)
    val_loader, _ = get_val_loader()

    evaluate(model, val_loader, class_names)
    show_sample_predictions(model, val_loader, class_names, n=8)


if __name__ == "__main__":
    main()
