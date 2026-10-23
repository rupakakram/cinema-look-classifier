import torch
import numpy as np
from PIL import Image
from model import CinemaClassifier
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CinemaClassifier().to(DEVICE)
checkpoint = torch.load("best_model.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Random noise image
noise = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
img = Image.fromarray(noise)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

tensor = transform(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]

print(f"cinematic: {probs[0].item():.1%}, normal: {probs[1].item():.1%}")