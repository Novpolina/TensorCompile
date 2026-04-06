#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "matrix.hpp"

template <typename T>
class Tensor final {
    std::size_t num_batches_, num_channels_, tensor_height_, tensor_width_; 
    std::vector<Matrix<T>> matrices_array_;

   public:
    using size_type = std::size_t;
    using value_type = Matrix<T>;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;

    Tensor(size_type batch_sz = 1, size_type channel_sz = 1, size_type h_size = 1, size_type w_size = 1)
        : num_batches_(batch_sz),
          num_channels_(channel_sz),
          tensor_height_(h_size),
          tensor_width_(w_size),
          matrices_array_(num_batches_ * num_channels_, Matrix<T>(tensor_height_, tensor_width_)) {}

    template <typename Iter>
        requires std::forward_iterator<Iter>
    Tensor(size_type batch_sz, size_type channel_sz, Iter it_start, Iter it_end)
        : num_batches_(batch_sz), num_channels_(channel_sz) {
        if (std::distance(it_start, it_end) != static_cast<difference_type>(batch_sz * channel_sz)) {
            throw std::invalid_argument("Mismatch between iterators size and new tensor size");
        }
        matrices_array_ = std::vector<value_type>(it_start, it_end);

        if (std::ranges::any_of(matrices_array_, [expected_size = matrices_array_[0].size()](const auto& matrix_elem) {
                return matrix_elem.size() != expected_size;
            })) {
            throw std::invalid_argument("Invalid initializer list matrix size");
        }

        tensor_height_ = matrices_array_[0].n_rows();
        tensor_width_ = matrices_array_[0].n_cols();
    }

    Tensor(size_type batch_sz, size_type channel_sz, std::initializer_list<value_type> init_list)
        : Tensor(batch_sz, channel_sz, init_list.begin(), init_list.end()) {}

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(Tensor&& other_tensor) noexcept
        : num_batches_(std::exchange(other_tensor.num_batches_, 0)),
          num_channels_(std::exchange(other_tensor.num_channels_, 0)),
          tensor_height_(std::exchange(other_tensor.tensor_height_, 0)),
          tensor_width_(std::exchange(other_tensor.tensor_width_, 0)),
          matrices_array_(std::move(other_tensor.matrices_array_)) {}

    Tensor& operator=(Tensor&& other_tensor) noexcept {
        matrices_array_ = std::move(other_tensor.matrices_array_);
        num_batches_ = std::exchange(other_tensor.num_batches_, 0);
        num_channels_ = std::exchange(other_tensor.num_channels_, 0);
        tensor_height_ = std::exchange(other_tensor.tensor_height_, 0);
        tensor_width_ = std::exchange(other_tensor.tensor_width_, 0);

        return *this;
    }

    ~Tensor() = default;

    constexpr value_type& operator[](size_type b_idx, size_type c_idx) { 
        return matrices_array_[b_idx * num_channels_ + c_idx]; 
    }

    constexpr value_type& at(size_type b_idx, size_type c_idx) {
        if (b_idx >= num_batches_ || c_idx >= num_channels_) {
            throw std::out_of_range("Tensor access fail");
        }
        return matrices_array_[b_idx * num_channels_ + c_idx];
    }

    constexpr const value_type& operator[](size_type b_idx, size_type c_idx) const {
        return matrices_array_[b_idx * num_channels_ + c_idx];
    }

    constexpr const value_type& at(size_type b_idx, size_type c_idx) const {
        if (b_idx >= num_batches_ || c_idx >= num_channels_) {
            throw std::out_of_range("Tensor access fail");
        }
        return matrices_array_[b_idx * num_channels_ + c_idx];
    }

    constexpr T& operator[](size_type b_idx, size_type c_idx, size_type y_idx, size_type x_idx) {
        return matrices_array_[b_idx * num_channels_ + c_idx][y_idx, x_idx];
    }

    constexpr T& at(size_type b_idx, size_type c_idx, size_type y_idx, size_type x_idx) {
        if (num_batches_ <= b_idx || num_channels_ <= c_idx || tensor_height_ <= y_idx || tensor_width_ <= x_idx) {
            throw std::out_of_range("Tensor access fail");
        }
        return matrices_array_[b_idx * num_channels_ + c_idx][y_idx, x_idx];
    }

    constexpr const T& operator[](size_type b_idx, size_type c_idx, size_type y_idx, size_type x_idx) const {
        return matrices_array_[b_idx * num_channels_ + c_idx][y_idx, x_idx];
    }

    constexpr const T& at(size_type b_idx, size_type c_idx, size_type y_idx, size_type x_idx) const {
        if (num_batches_ <= b_idx || num_channels_ <= c_idx || tensor_height_ <= y_idx || tensor_width_ <= x_idx) {
            throw std::out_of_range("Tensor access fail");
        }
        return matrices_array_[b_idx * num_channels_ + c_idx][y_idx, x_idx];
    }

    size_type batch() const noexcept { return num_batches_; }
    size_type channels() const noexcept { return num_channels_; }
    size_type height() const noexcept { return tensor_height_; }
    size_type width() const noexcept { return tensor_width_; }

    size_type num_elements() const noexcept { return num_matrices() * tensor_height_ * tensor_width_; }
    size_type num_matrices() const noexcept { return matrices_array_.size(); }
    constexpr auto& data() noexcept { return matrices_array_; }
    constexpr const auto& data() const noexcept { return matrices_array_; }

    void print() const {
        for (size_type b_idx = 0; b_idx < num_batches_; ++b_idx) {
            std::cout << "Batch " << b_idx << ":\n";
            for (size_type c_idx = 0; c_idx < num_channels_; ++c_idx) {
                std::cout << " Channel " << c_idx << ":\n";
                for (size_type y_idx = 0; y_idx < tensor_height_; ++y_idx) {
                    for (size_type x_idx = 0; x_idx < tensor_width_; ++x_idx) {
                        std::cout << (*this)[b_idx, c_idx, y_idx, x_idx] << ' ';
                    }
                    std::cout << '\n';
                }
            }
        }
    }
};