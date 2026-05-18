import torch
import torchvision.models as models
import numpy as np
import os

os.makedirs("data", exist_ok=True)

print("Загрузка SqueezeNet...")
model = models.squeezenet1_1(pretrained=False)
model.eval()

with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.bias = None 
            module.weight.fill_(0.01)

dummy_x = torch.full((1, 3, 224, 224), 1.0)

print("Сохранение входных данных для C++ Runner'а...")
dummy_x.numpy().astype(np.float32).tofile("data/input_224.bin")

print("Экспорт в ONNX...")
torch.onnx.export(
    model, 
    dummy_x, 
    "data/squeezenet.onnx",
    input_names=["input_X"],
    output_names=["output_C"],
    opset_version=14
)

with torch.no_grad():
    pytorch_out = model(dummy_x)

pytorch_out.numpy().astype(np.float32).tofile("data/output_ref.bin")
    
print("\nPyTorch эталон (первые 5 элементов из 1000):")
print(pytorch_out[0][:5].numpy())