from MobileNetV2 import mobilenet_v2
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

net = mobilenet_v2(pretrained=True)
net.eval()

image_path = './temp_data/test_run.png'
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = net(image_tensor)

print(output.shape)
binary_mask = (output > 0.5).float()
binary_mask_np = binary_mask.cpu().squeeze().numpy()
binary_mask_np = (binary_mask_np * 255).astype(np.uint8)

binary_mask_image = Image.fromarray(binary_mask_np)
binary_mask_image.save('./temp_data/temp_mask.png')