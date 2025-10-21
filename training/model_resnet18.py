import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_modified_resnet18():
    print("🧠 Loading pretrained ResNet18...")
    base_model = models.resnet18(pretrained=True)

    # Customizing conv1 and fc layer
    base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base_model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base_model.fc.in_features, 1),
        nn.Sigmoid()
    )

    base_model.to(DEVICE)
    return base_model

def summarize_model(model, input_size=(3, 224, 224)):
    print("\n📋 Model Summary:")
    print(f"{'Index':<6} {'Layer Type':<20} {'Input Shape':<25} {'Output Shape':<25} {'Param #':<10}")
    print("-" * 90)
    
    summary = []
    hooks = []

    def register_hook(module):
        def hook(module, input, output):
            layer_idx = len(summary)
            layer_type = module.__class__.__name__
            input_shape = tuple(input[0].size()) if isinstance(input, tuple) else tuple(input.size())
            output_shape = tuple(output.size()) if isinstance(output, torch.Tensor) else str(output)
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary.append((layer_idx, layer_type, input_shape, output_shape, num_params))
        return hook

    # Register hook to all layers except containers
    for layer in model.modules():
        if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and not (layer == model):
            hooks.append(layer.register_forward_hook(register_hook(layer)))

    dummy_input = torch.randn(1, *input_size).to(DEVICE)
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    for layer in summary:
        idx, layer_type, input_shape, output_shape, num_params = layer
        print(f"{idx:<6} {layer_type:<20} {str(input_shape):<25} {str(output_shape):<25} {num_params:<10}")

if __name__ == "__main__":
    model = get_modified_resnet18()
    summarize_model(model)
