#include <iostream>
#include <vector>
#include <numeric>
#include <cstdint>
#include <cstdlib>

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
    MemRefDescriptor<float, 4>* input_X,
    MemRefDescriptor<float, 4>* input_Add,
    MemRefDescriptor<float, 4>* input_Mul
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

int main() {
    std::vector<float> data_x(1 * 3 * 32 * 32, 1.0f);
    std::vector<float> data_add(1 * 16 * 16 * 16, 0.5f);
    std::vector<float> data_mul(1 * 16 * 16 * 16, 2.0f);

    MemRefDescriptor<float, 4> desc_x, desc_add, desc_mul;
    MemRefDescriptor<float, 2> desc_out;

    init_descriptor(desc_x, data_x, {1, 3, 32, 32});
    init_descriptor(desc_add, data_add, {1, 16, 16, 16});
    init_descriptor(desc_mul, data_mul, {1, 16, 16, 16});

    std::cout << "Running compiled tensor function...\n";

    _mlir_ciface_forward(&desc_out, &desc_x, &desc_add, &desc_mul);

    std::cout << "Output tensor (first 5 elements): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << desc_out.aligned[i] << " ";
    }
    std::cout << std::endl;

    free(desc_out.allocated);

    return 0;
}