import torch
import torch.nn as nn
import os

os.makedirs("data", exist_ok=True)

class FullAssignmentModel(nn.Module):
    def __init__(self):
        super(FullAssignmentModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.gemm_layer = nn.Linear(4096, 128, bias=False)
        self.matmul_weight = nn.Parameter(torch.ones(128, 10))

    def forward(self, x, add_tensor, mul_tensor):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.add(x, add_tensor)
        x = torch.mul(x, mul_tensor)
        x = torch.flatten(x, 1)
        x = self.gemm_layer(x)
        out = torch.matmul(x, self.matmul_weight)
        return out

model = FullAssignmentModel()

with torch.no_grad():
    for param in model.parameters():
        param.fill_(1.0)

model.eval()

dummy_x = torch.full((1, 3, 33, 33), 1.0)
dummy_add = torch.full((1, 16, 16, 16), 0.5)
dummy_mul = torch.full((1, 16, 16, 16), 2.0)

torch.onnx.export(
    model, 
    (dummy_x, dummy_add, dummy_mul), 
    "data/full_model.onnx",
    input_names=["input_X", "input_Add", "input_Mul"],
    output_names=["output_C"],
    opset_version=14
)

print("Полная модель успешно сохранена в data/full_model.onnx")

with torch.no_grad():
    pytorch_out = model(dummy_x, dummy_add, dummy_mul)
    
print("PyTorch output (first 5 elements):")
print(pytorch_out[0][:5].numpy())