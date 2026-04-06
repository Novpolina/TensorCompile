#pragma once

#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
class Matrix final : private std::vector<T> {
   private:
    std::size_t num_rows_;
    std::size_t num_cols_;

   public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;
    using reverse_iterator = typename std::vector<value_type>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<value_type>::const_reverse_iterator;
    using iterator_category = typename std::iterator_traits<iterator>::iterator_category;

    using std::vector<value_type>::data;

    Matrix(size_type r_count, size_type c_count)
        : std::vector<T>(r_count * c_count), num_rows_(r_count), num_cols_(c_count) {}

    Matrix(size_type r_count, size_type c_count, std::initializer_list<value_type> init_list)
        : num_rows_(r_count), num_cols_(c_count) {
        if (init_list.size() != num_rows_ * num_cols_) {
            throw std::invalid_argument("Incorrect initializer list size");
        }
        this->assign(init_list);
    }

    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;

    Matrix(Matrix&& other_mat) noexcept
        : std::vector<value_type>(std::move(other_mat)),
          num_rows_(std::exchange(other_mat.num_rows_, 0)),
          num_cols_(std::exchange(other_mat.num_cols_, 0)) {}

    Matrix& operator=(Matrix&& other_mat) noexcept {
        std::vector<T>::operator=(std::move(other_mat));
        num_rows_ = std::exchange(other_mat.num_rows_, 0);
        num_cols_ = std::exchange(other_mat.num_cols_, 0);
        return *this;
    }

    ~Matrix() = default;

    constexpr value_type& operator[](size_type r_idx, size_type c_idx) {
        return data()[r_idx * num_cols_ + c_idx];
    }

    constexpr value_type& at(size_type r_idx, size_type c_idx) {
        if (!(r_idx < num_rows_ && c_idx < num_cols_)) {
            throw std::invalid_argument("Invalid matrix access");
        }
        return data()[r_idx * num_cols_ + c_idx];
    }

    constexpr const value_type& operator[](size_type r_idx, size_type c_idx) const {
        return data()[r_idx * num_cols_ + c_idx];
    }

    constexpr const value_type& at(size_type r_idx, size_type c_idx) const {
        if (!(r_idx < num_rows_ && c_idx < num_cols_)) {
            throw std::invalid_argument("Invalid matrix access");
        }
        return data()[r_idx * num_cols_ + c_idx];
    }

    size_type n_cols() const noexcept { return num_cols_; }
    size_type n_rows() const noexcept { return num_rows_; }
    using std::vector<value_type>::size;

    using std::vector<value_type>::begin;
    using std::vector<value_type>::end;
    using std::vector<value_type>::cbegin;
    using std::vector<value_type>::cend;
    using std::vector<value_type>::rbegin;
    using std::vector<value_type>::rend;
    using std::vector<value_type>::crbegin;
    using std::vector<value_type>::crend;

    bool equal(const Matrix& compare_mat) const { return data() == compare_mat.data(); }

    void transpose() {
        Matrix result_transposed(num_cols_, num_rows_);
        for (std::size_t r = 0; r < num_rows_; ++r) {
            for (std::size_t c = 0; c < num_cols_; ++c) {
                result_transposed[c, r] = (*this)[r, c];
            }
        }
        std::swap(*this, result_transposed);
    }
};

template <typename T>
Matrix<T> operator*(Matrix<T>& lhs_mat, Matrix<T>& rhs_mat) {
    const std::size_t right_c = rhs_mat.n_cols();
    const std::size_t right_r = rhs_mat.n_rows();
    const std::size_t left_c = lhs_mat.n_cols();
    const std::size_t left_r = lhs_mat.n_rows();

    if (left_c != right_r) {
        throw std::invalid_argument("Invalid matrix size");
    }

    Matrix<T> result_mat(left_r, right_c);

    for (std::size_t r = 0; r < left_r; ++r) {
        for (std::size_t c = 0; c < right_c; ++c) {
            for (std::size_t k_idx = 0; k_idx < left_c; ++k_idx) {
                result_mat[r, c] += lhs_mat[r, k_idx] * rhs_mat[k_idx, c];
            }
        }
    }

    return result_mat;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs_mat, const Matrix<T>& rhs_mat) {
    const std::size_t right_c = rhs_mat.n_cols();
    const std::size_t right_r = rhs_mat.n_rows();
    const std::size_t left_c = lhs_mat.n_cols();
    const std::size_t left_r = lhs_mat.n_rows();

    if (left_c != right_r) {
        throw std::invalid_argument("Invalid matrix size");
    }

    Matrix<T> result_mat(left_r, right_c);

    for (std::size_t r = 0; r < left_r; ++r) {
        for (std::size_t c = 0; c < right_c; ++c) {
            for (std::size_t k_idx = 0; k_idx < left_c; ++k_idx) {
                result_mat[r, c] += lhs_mat[r, k_idx] * rhs_mat[k_idx, c];
            }
        }
    }

    return result_mat;
}