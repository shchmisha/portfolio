//
// Created by Миша Щербаков on 18.07.2022.
//

#ifndef NEURALNETWORK_MATRIX_H
#define NEURALNETWORK_MATRIX_H

class Matrix{
public:
    int rows;
    int cols;
    double **entries;
    Matrix(int rows, int cols);
    Matrix(const Matrix &source);
    Matrix(Matrix &&source);
    Matrix subset(int fromRow, int toRow, int toCol, int fromCol);
    void fill(double n);
    void print();
    Matrix &operator=(const Matrix &rhs);
    Matrix &operator=(Matrix &&m);
    Matrix operator+(const Matrix &rhs) const;
    Matrix operator*(const Matrix &rhs) const;
    Matrix operator-(const Matrix &rhs) const;
    Matrix dot(const Matrix &m) const;
    Matrix transpose() const;
    Matrix apply(double (*func)(double));
    void randomize(double fMin, double fMax);
    static Matrix ones(int rows, int cols);
    static Matrix zeros(int rows, int cols);
    Matrix scale(double x);
    ~Matrix();
};

Matrix crossCorrelation(Matrix &m1, Matrix &m2);

#endif //NEURALNETWORK_MATRIX_H
