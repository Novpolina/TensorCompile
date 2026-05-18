#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

template <typename T, size_t Rank>
struct MemRefDescriptor {
    T *allocated;
    T *aligned;
    intptr_t offset;
    intptr_t sizes[Rank];
    intptr_t strides[Rank];
};

extern "C" void _mlir_ciface_forward(
    MemRefDescriptor<float, 2>* output_C, 
    MemRefDescriptor<float, 4>* input_X   
);

template <size_t Rank>
void init_descriptor(MemRefDescriptor<float, Rank>& desc, std::vector<float>& data, std::vector<intptr_t> dims) {
    desc.allocated = data.data();
    desc.aligned = data.data();
    desc.offset = 0;
    for (size_t i = 0; i < Rank; ++i) desc.sizes[i] = dims[i];
    
    intptr_t current_stride = 1;
    for (int i = Rank - 1; i >= 0; --i) {
        desc.strides[i] = current_stride;
        current_stride *= dims[i];
    }
}

std::vector<float> load_binary(const std::string& filepath, size_t expected_size) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open " + filepath);
    
    std::vector<float> data(expected_size);
    file.read(reinterpret_cast<char*>(data.data()), expected_size * sizeof(float));
    return data;
}

int main() {
    try {
        std::cout << "Loading PyTorch inputs...\n";
        std::vector<float> data_x = load_binary("data/input_224.bin", 150528);

        MemRefDescriptor<float, 4> desc_x;
        MemRefDescriptor<float, 2> desc_out; 

        init_descriptor(desc_x, data_x, {1, 3, 224, 224});

        std::cout << "Running AOT compiled SqueezeNet...\n";
        _mlir_ciface_forward(&desc_out, &desc_x);

        std::cout << "SqueezeNet AOT output (first 5 elements): ";
        for (int i = 0; i < 5; ++i) {
            std::cout << desc_out.aligned[i] << " ";
        }
        std::cout << "\n";

        std::vector<float> ref_out = load_binary("data/output_ref.bin", 1000);
        
        std::cout << "PyTorch Reference     (first 5 elements): ";
        for (int i = 0; i < 5; ++i) {
            std::cout << ref_out[i] << " ";
        }
        std::cout << "\n";
        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        int mismatch_count = 0;
        float tolerance = 0.5;

        for (size_t i = 0; i < 1000; ++i) {
            float diff = std::abs(desc_out.aligned[i] - ref_out[i]);
            sum_diff += diff;
            
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > tolerance) {
                mismatch_count++;
            }
        }

        float mae = sum_diff / 1000.0f; 

        std::cout << "===========================================\n";
        std::cout << "РЕЗУЛЬТАТЫ ПОЛНОЙ ВЕРИФИКАЦИИ (1000 классов):\n";
        std::cout << "Максимальное абсолютное расхождение (Max Error): " << max_diff << "\n";
        std::cout << "Среднее абсолютное расхождение (MAE): " << mae << "\n";
        
        if (mismatch_count == 0) {
            std::cout << "УСПЕХ! Все 1000 элементов совпали в пределах допуска " << tolerance << ".\n";
        } else {
            std::cout << "ВНИМАНИЕ! Найдено " << mismatch_count << " элементов с расхождением > " << tolerance << "\n";
        }
        std::cout << "===========================================\n";

        free(desc_out.allocated);
        
        std::cout << "Execution finished successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << "\n";
    }
    return 0;
}