import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import CinemaClassifier
from dataset import get_loaders

# Configurations
DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
SAVE_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def plot_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train loss", color="#2563eb")
    ax1.plot(val_losses, label="Val loss", color="#dc2626")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot([a * 100 for a in train_accs], label="Train acc", color="#2563eb")
    ax2.plot([a * 100 for a in val_accs], label="Val acc", color="#dc2626")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    plt.show()
    print("Saved: training_curves.png")

def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, class_names = get_loaders(DATA_DIR, BATCH_SIZE)

    model = CinemaClassifier(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Reduce LR when val_acc plateaus (stop improving)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("_" * 58)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        marker = " saved" if val_acc > best_val_acc else ""
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.2%} | {val_loss:>8.4f} | {val_acc:>7.2%}{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "classes": class_names,
            }, SAVE_PATH)

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2%}")
    print(f"Best model saved to: {SAVE_PATH}")

    plot_curves(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    main()
