#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>

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

std::istream &operator>>(std::istream &in, Matrix &matrix) {
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix.size(); ++j) {
            in >> matrix[matrix.Index(i, j)];
        }
    }
    return in;
}

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
    int matrix_size = 1;
    bool parallelize_inner = false, parallelize_outer = false;
    bool use_block = false;
    int block_size = 1;
    std::vector<std::string> options = {"-i", "-o", "-s", "-b"};
    for (int i = 1; i < argc; ++i) {
        for (auto &option : options) {
            if (strcmp(option.c_str(), argv[i]) == 0) {
                if (option == "-s") {
                    if (i + 1 == argc) {
                        throw std::runtime_error("You must specify matrix matrix_size after -s");
                    } else {
                        matrix_size = std::stoi(argv[i + 1]);
                    }
                } else if (option == "-i") {
                    parallelize_inner = true;
                } else if (option == "-o") {
                    parallelize_outer = true;
                } else if (option == "-b") {
                    use_block = true;
                    if (i + 1 == argc) {
                        throw std::runtime_error("You must specify block matrix_size after -b");
                    } else {
                        block_size = std::stoi(argv[i + 1]);
                    }
                }
            }
        }
    }
    Matrix A(std::vector<int>(matrix_size * matrix_size, 0));
    Matrix B(std::vector<int>(matrix_size * matrix_size, 0));
    std::cin >> A >> B;
    if (!use_block) {
        Matrix C = NonBlockMultiplication(A, B, parallelize_outer, parallelize_inner);
        std::cout << C;
    } else {
        Matrix C = BlockMultiplication(A, B, block_size, parallelize_outer, parallelize_inner);
        std::cout << C;
    }
    return 0;
}
