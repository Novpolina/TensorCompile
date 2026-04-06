import torch
import torch.nn as nn
import os

os.makedirs("data", exist_ok=True)

class FullAssignmentModel(nn.Module):
    def __init__(self):
        super(FullAssignmentModel, self).__init__()
        # 1. Conv: Сверточный слой (in_channels=3, out_channels=16)
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        
        # 2. Relu: Функция активации
        self.relu = nn.ReLU()
        
        # 3. Gemm: Полносвязный слой со смещением (bias=True) экспортируется в Gemm
        # Размер входа после свертки (32x32 -> 16x16) = 16 каналов * 16 * 16 = 4096
        self.gemm_layer = nn.Linear(4096, 128, bias=True)
        
        # 4. MatMul: Для этой операции мы будем использовать простое перемножение матриц
        self.matmul_weight = nn.Parameter(torch.randn(128, 10))

    def forward(self, x, add_tensor, mul_tensor):
        # --- Блок свертки ---
        x = self.conv(x)                  # Операция Conv
        x = self.relu(x)                  # Операция Relu
        
        # --- Поэлементные операции ---
        x = torch.add(x, add_tensor)      # Операция Add
        x = torch.mul(x, mul_tensor)      # Операция Mul
        
        # Разворачиваем тензор (Flatten) в 2D для линейных слоев
        # Flatten не указан в задании, но нужен для стыковки Conv и MatMul
        x = torch.flatten(x, 1)           
        
        # --- Матричные операции ---
        x = self.gemm_layer(x)            # Операция Gemm
        out = torch.matmul(x, self.matmul_weight) # Операция MatMul
        
        return out

model = FullAssignmentModel()
model.eval()

# Создаем входные тензоры с нужными размерностями
# Изображение: Batch=1, Channels=3, Height=32, Width=32
dummy_x = torch.randn(1, 3, 32, 32)

# Тензоры для Add и Mul (должны совпадать с выходом Conv: 1, 16, 16, 16)
dummy_add = torch.randn(1, 16, 16, 16)
dummy_mul = torch.randn(1, 16, 16, 16)

# Экспорт модели
torch.onnx.export(
    model, 
    (dummy_x, dummy_add, dummy_mul), 
    "data/full_model.onnx",
    input_names=["input_X", "input_Add", "input_Mul"],
    output_names=["output_C"],
    opset_version=14 # Используем стабильную версию набора операций ONNX
)

print("Полная модель успешно сохранена в data/full_model.onnx")