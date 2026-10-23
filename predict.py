import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import CinemaClassifier

# Configuration
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["classes"]
    model = CinemaClassifier(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return img, tensor

def predict(image_path, model, class_names):
    img, tensor = preprocess(image_path)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    all_probs = {class_names[i]: probs[i].item() for i in range(len(class_names))}

    return pred_class, confidence, all_probs, img


def show_result(pred_label, confidence, all_probs, img):
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: the image
    ax_img.imshow(img)
    ax_img.axis('off')
    color = "#16a34a" if pred_label == 'cinematic' else "#dc2626"
    ax_img.set_title(
        f"{pred_label.upper()} {confidence:.1%}", fontsize=14, fontweight="bold", color=color, pad=10
    )

    # Right: the probabilitiy bar chart
    labels = list(all_probs.keys())
    values = [v * 100 for v in all_probs.values()]
    colors = ["#2563eb" if l == pred_label else "#cbd5e1" for l in labels]

    bars = ax_bar.barh(labels, values, color=colors, height=0.4)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Confidence (%)")
    ax_bar.set_title("Class Probabilities")
    ax_bar.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values):
        ax_bar.text(val +1, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=11)
        
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=300)
    plt.show()
    print("Saved: prediction_result.png")

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to the image: ").strip()

    if not image_path:
        print("No image path provided.")
        return
    
    print(f"\nLoading model from {MODEL_PATH}")
    model, class_names = Load_model(MODEL_PATH)

    print(f"Running prediction on {image_path}\n")
    pred_label, confidence, all_probs, img = predict(image_path, model, class_names)

    print("_" * 40)
    print(f"Prediction : {pred_label.upper()}")
    print(f"Confidence : {confidence:.1%}")
    print("_" * 40)
    for label, prob in all_probs.items():
        bar = "█" * int(prob * 30)
        print(f"{label:<12} {bar:<30} {prob:.1%}")
    print("_" * 40)

    show_result(pred_label, confidence, all_probs, img)


if __name__ == "__main__":
    main()
