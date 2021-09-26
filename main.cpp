#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

class Matrix {
public:
    Matrix(std::vector<int> values) : values_(std::move(values)) {
        size_ = std::sqrt(values_.size());
        if (size_ * size_ != values_.size()) {
            throw std::runtime_error("You must enter square matrices");
        }
    }

    int &operator[](int index) {
        return values_[index];
    }

    int operator[](int index) const {
        return values_[index];
    }

    [[nodiscard]] int Index(int i, int j) const {
        return i * size_ + j;
    }

    [[nodiscard]] int size() const {
        return size_;
    }

private:
    std::vector<int> values_;
    int size_;
};

std::ostream &operator<<(std::ostream &out, const Matrix &matrix) {
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix.size(); ++j) {
            out << matrix[matrix.Index(i, j)] << " ";
        }
        out << "\n";
    }
    return out;
}


Matrix NonBlockMultiplication(const Matrix &A, const Matrix &B, bool parallelize_outer = false,
                              bool parallelize_inner = false) {
    if (A.size() != B.size()) {
        throw std::runtime_error("You must multiply matrices of the same dimension");
    }
    Matrix C(std::vector<int>(A.size() * A.size()));
    int i;
    #pragma omp parallel for shared(A, B, C, parallelize_inner) private(i) if(parallelize_outer) default(none)
    for (i = 0; i < C.size(); ++i) {
        int j;
        #pragma omp parallel for shared(i, A, B, C) private(j) if(parallelize_inner) default(none)
        for (j = 0; j < C.size(); ++j) {
            for (int k = 0; k < C.size(); ++k) {
                C[C.Index(i, j)] += A[A.Index(i, k)] * B[B.Index(k, j)];
            }
        }
    }
    return C;
}

Matrix BlockMultiplication(const Matrix &A, const Matrix &B, int block_size, bool parallelize_outer = false,
                           bool parallelize_inner = false) {
    if (A.size() != B.size()) {
        throw std::runtime_error("You must multiply matrices of the same dimension");
    }
    if (A.size() % block_size != 0) {
        throw std::runtime_error("Block size must be a divisor of matrix size");
    }
    Matrix C(std::vector<int>(A.size() * A.size()));
    int n_blocks = A.size() / block_size;
    int i, j;
    #pragma omp parallel for shared(n_blocks, block_size, A, B, C, parallelize_inner) private(i) if(parallelize_outer) default(none)
    for (i = 0; i < n_blocks; ++i) {
        #pragma omp parallel for shared(i, n_blocks, block_size, A, B, C) private(j) if(parallelize_inner) default(none)
        for (j = 0; j < n_blocks; ++j) {
            for (int k = 0; k < n_blocks; ++k) {
                for (int i_block = i * block_size; i_block < (i + 1) * block_size; ++i_block) {
                    for (int j_block = j * block_size; j_block < (j + 1) * block_size; ++j_block) {
                        for (int k_block = k * block_size; k_block < (k + 1) * block_size; ++k_block) {
                            C[C.Index(i_block, j_block)] += A[A.Index(i_block, k_block)] * B[B.Index(k_block, j_block)];
                        }
                    }
                }
            }
        }
    }
    return C;
}


int main(int argc, char *argv[]) {
    int size = 10;
    Matrix A(std::vector<int>(size * size, 1));
    Matrix B(std::vector<int>(size * size, 1));
    auto start = std::chrono::steady_clock::now();
    std::cout << BlockMultiplication(A, B, 1, true, false);
    auto mid = std::chrono::steady_clock::now();
    std::cout << BlockMultiplication(A, B, 1, false, true);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count() << " "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - mid).count();
    return 0;
}
