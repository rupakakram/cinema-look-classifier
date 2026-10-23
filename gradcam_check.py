import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from model import CinemaClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    checkpoint = torch.load("best_model.pth", map_location=DEVICE, weights_only=False)
    class_names = checkpoint["classes"]
    model = CinemaClassifier(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names

def gradcam(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    input_tensor = tensor.requires_grad_()

    activations = {}
    gradients = {}

    last_conv_layer = model.features[-4]

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = last_conv_layer.register_forward_hook(forward_hook)
    h2 = last_conv_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_label = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    model.zero_grad()
    output[0, pred_idx].backward()

    h1.remove()
    h2.remove()

    acts = activations["value"][0]
    grads = gradients["value"][0]

    weights = torch.mean(grads, dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], device=DEVICE)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().detach().numpy()

    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
    cam_resized = np.array(cam_resized) / 255.0

    # Prepare the original image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display_img = input_tensor[0].detach().cpu() * std + mean
    display_img = display_img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(display_img)
    axes[0].set_title(f"Input\nPred: {pred_label} ({confidence:.1f})")
    axes[0].axis("off")

    axes[1].imshow(display_img)
    axes[1].imshow(cam_resized, cmap="jet", alpha=0.5)
    axes[1].set_title("Grad-CAM: what the model focoused on")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=300)
    plt.show()
    print(f"Prediction: {pred_label} ({confidence:.1f})")
    print("Saved: gradcam_result.png")

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to the image: ").strip()

    model, class_names = load_model()
    gradcam(model, image_path, class_names)



if __name__ == "__main__":
    main()
