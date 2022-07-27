//
// Created by Миша Щербаков on 18.07.2022.
//

#include "Matrix.h"
#include "iostream"

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->entries = new double*[rows];
    for (int i = 0; i < rows; i++){
        this->entries[i] = new double[cols];
    }
}

void Matrix::fill(double n) {
    for (int i = 0; i<rows;i++){
        for (int j = 0; j<cols; j++){
            this->entries[i][j] = n;
        }
    }
}

Matrix::~Matrix() {
    for (int i = 0; i<rows;i++){
        free(entries[i]);
    }
    free(entries);
}

void Matrix::print() {
    for (int i = 0; i<rows;i++){
        for (int j = 0; j<cols; j++){
            std::cout << this->entries[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::subset(int fromRow, int toRow, int fromCol, int toCol) {
//    std::cout << "passed" << toRow << " " << fromRow << " " << toCol << " " << fromCol<< std::endl;
    Matrix mat = Matrix(toRow-fromRow, toCol-fromCol);
//    std::cout << "passed"<<std::endl;
    for (int i = fromRow;i<toRow;i++){
        for (int j = fromCol; j<toCol;j++){
            mat.entries[i-fromRow][j-fromCol] = this->entries[i][j];
        }
    }
    return mat;
}

Matrix &Matrix::operator=(const Matrix &rhs) {
    if (this==&rhs)
        return *this;

    for (int i = 0; i<rows;i++)
        delete [] entries[i];

    delete [] entries;

    rows = rhs.rows;
    cols = rhs.cols;

    for (int i = 0; i < rows; i++){
        this->entries[i] = new double[cols];
    }

    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            this->entries[i][j] = rhs.entries[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix &rhs) const {
    if (cols==rhs.cols && rows == rhs.rows){
        Matrix m {rows,cols};
        for (int i=0;i<m.rows;i++){
            for (int j=0;j<m.cols;j++){
                m.entries[i][j]=m.entries[i][j] + rhs.entries[i][j];
            }
        }
        return m;
    } else {
        std::cout << "Dimensions must be the same when adding" << std::endl;
        exit(1);
    }
}

Matrix Matrix::operator*(const Matrix &rhs) const {
    if (cols==rhs.cols && rows == rhs.rows){
        Matrix m = Matrix(rows, cols);
        for (int i=0;i<m.rows;i++){
            for (int j=0;j<m.cols;j++){
                m.entries[i][j]=entries[i][j] * rhs.entries[i][j];
            }
        }
        return m;
    } else {
        std::cout << "Dimensions must be the same when multiplying" << std::endl;
        exit(1);
    }
}

Matrix Matrix::operator-(const Matrix &rhs) const {
    if (cols==rhs.cols && rows == rhs.rows){
        Matrix m = Matrix(rows, cols);
        for (int i=0;i<m.rows;i++){
            for (int j=0;j<m.cols;j++){
                m.entries[i][j]=m.entries[i][j] - rhs.entries[i][j];
            }
        }
        return m;
    } else {
        std::cout << "Dimensions must be the same when subtracting" << std::endl;
        exit(1);
    }
}

Matrix Matrix::dot(const Matrix &m) const {
    if (cols == m.rows){
        Matrix m = Matrix(rows, m.cols);
        for (int i=0;i<rows;i++){
            for (int j=0;j<m.cols;j++){
                double sum =0;
                for (int k=0; k<m.rows;k++){
                    sum+=entries[i][k]*m.entries[k][j];
                }
                m.entries[i][j]=sum;
            }
        }
        return m;
    } else {
        std::cout << "Dimensions must be the same when dotting" << std::endl;
        exit(1);
    }
}

Matrix Matrix::transpose() const {
    Matrix mat = Matrix(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat.entries[j][i] = entries[i][j];
        }
    }
    return mat;
}

Matrix Matrix::apply(double (*func)(double)) {
    Matrix mat{rows, cols};
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            mat.entries[i][j]=(*func)(entries[i][j]);
        }
    }
    return mat;
}

void Matrix::randomize(double fMin, double fMax) {
    for (int i = 0; i<rows;i++){
        for (int j = 0; j<cols; j++){
            double f = (double)rand() / RAND_MAX;
            entries[i][j] = f;
        }
    }
}

Matrix &Matrix::operator=(Matrix &&m) {
    if (this==&m)
        return *this;

    rows = m.rows;
    cols = m.cols;

    for (int i = 0;i<rows; i++){
        delete [] entries[i];
    }

    delete [] entries;

    entries = m.entries;
    m.entries = nullptr;

    return *this;
}

Matrix::Matrix(const Matrix &source) {
    rows = source.rows;
    cols = source.cols;
    entries = new double*[rows];
    for (int i = 0; i<rows; i++){
        entries[i] = new double[cols];
        for (int j = 0; j<cols;j++){
            entries[i][j] = source.entries[i][j];
        }
    }
}

Matrix::Matrix(Matrix &&source) {
    rows = source.rows;
    cols = source.cols;
    entries = source.entries;
    source.entries = nullptr;
}

Matrix Matrix::ones(int rows, int cols) {
    Matrix mat {rows, cols};
    for (int i = 0; i< rows; i++){
        for (int j=0; j<cols;j++){
            mat.entries[i][j] = 1;
        }
    }
    return mat;
}

Matrix Matrix::zeros(int rows, int cols) {
    Matrix mat {rows, cols};
    for (int i = 0; i< rows; i++){
        for (int j=0; j<cols;j++){
            mat.entries[i][j] = 0;
        }
    }
    return mat;
}

Matrix Matrix::scale(double x) {
    Matrix mat {rows, cols};
    for (int i = 0; i< rows; i++){
        for (int j=0; j<cols;j++){
            mat.entries[i][j] = entries[i][j] * x;
        }
    }
    return mat;
}

//Matrix crossCorrelation(Matrix &m1, Matrix &m2){
//    Matrix mat = Matrix(m1.rows-m2.rows+1, m1.cols-m2.cols+1);
//
//    for (int i = 0; i<mat.rows; i++){
//        // problem is that the correlation funcition is counting th ecoordinates of matrix differently
//        // ACCOUNT FOR ROW OFFSET
//        for (int j = 0; j<mat.cols; j++){
//            double sum = 0.0;
//
//            Matrix subMat = m1.subset(i, i+m2.rows, j, j+m2.cols);
////            std::cout << "passed" << std::endl;
//
////            subMat.print();
////            std::cout << "passed" << std::endl;
//
////            m2.print();
//            Matrix product = multiply(subMat, *m2);
////            if (i==0 && j==0){
////                subMat.print();
////                m2.print();
////                multiply(subMat, m2);
////            }
//
//            for (int m = 0; m<product.rows; m++){
//                for (int n = 0; n<product.cols; n++){
////                    std::cout << m << " " << n << " " << m-i << " " << n-j << " "  <<  std::endl;
//                    sum += product.entries[m][n];
//                }
//            }
//
////            std::cout << sum << std::endl;
////            mat.print();
//            mat.entries[i][j] = sum;
////            mat.print();
//
//        }
//    }
//    return mat;
//}
//
//Matrix ones(int rows, int cols){
//    Matrix mat = Matrix(rows, cols);
//    for (int i=0;i<rows;i++){
//        for (int j=0;j<cols;j++){
//            mat.entries[i][j] = 1;
//        }
//    }
//    return mat;
//}
//
//Matrix zeros(int rows, int cols){
//    Matrix mat = Matrix(rows, cols);
//    for (int i=0;i<rows;i++){
//        for (int j=0;j<cols;j++){
//            mat.entries[i][j] = 0;
//        }
//    }
//    return mat;
//}
//
//
//
