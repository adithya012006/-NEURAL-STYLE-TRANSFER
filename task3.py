import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Function to load and resize image
def load_image(img_path, max_size=512, shape=None):
    image = Image.open(img_path).convert('RGB')
    if shape is not None and isinstance(shape, tuple):
        image = image.resize(shape)
    else:
        size = max_size if max(image.size) > max_size else max(image.size)
        image = image.resize((size, size))
    transform = transforms.ToTensor()
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

# Load your images (use the exact names of your uploaded images)
content = load_image("smile.jpg", max_size=256)
shape = (content.shape[-1], content.shape[-2])  # Note: PIL expects (width, height)
style = load_image("styles.jpg", shape=shape)

# Load pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features

# Freeze VGG parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content = content.to(device)
style = style.to(device)
vgg.to(device)

# Extract specific layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '10': 'conv3_1', '19': 'conv4_1',
                  '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix for style
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# Extract features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create the target image
target = content.clone().requires_grad_(True).to(device)

# Weights for style layers
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}
content_weight = 1
style_weight = 1e6

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training loop
for i in range(1, 101):  # 300 iterations
    target_features = get_features(target, vgg)

    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_weights:
        target_feat = target_features[layer]
        target_gram = gram_matrix(target_feat)
        style_gram = style_grams[layer]
        layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_loss

    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Print loss every 50 steps
    if i % 50 == 0:
        print(f"Iteration {i}, Total loss: {total_loss.item():.4f}")

# Convert tensor to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

# Show the result
plt.figure(figsize=(10, 5))
plt.imshow(im_convert(target))
plt.title("Stylized Image")
plt.axis("off")
plt.show()

# Optional: Save the result
from torchvision.utils import save_image
save_image(target, "stylized_output.jpg")
print("Image saved as stylized_output.jpg")
