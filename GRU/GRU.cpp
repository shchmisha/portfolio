//
// Created by Миша Щербаков on 21.07.2022.
//

#include "GRU.h"

GRU::GRU(int input, int hidden, int output) : Wrx(hidden, input), Wrh(hidden, hidden), Wzx(hidden, input), Wzh(hidden, hidden), Whx(hidden, input), Whh(hidden, hidden) {
    // randomize the matricies
    hiddenDims = hidden;
    inputDims = input;
    outputDims = output;
}

Matrix GRU::calcResetGate(Matrix &input, Matrix &prevState) {
    Matrix res = Wrx.dot(input) + Wrh.dot(prevState);
    return res.apply(activation);
}

Matrix GRU::calcUpdateGate(Matrix &input, Matrix &prevState) {
    Matrix res = Wzx.dot(input) + Wzh.dot(prevState);
    return res.apply(activation);
}

Matrix GRU::calcCandidateHiddenState(Matrix &input, Matrix &prevState, Matrix &ResetGate) {
    Matrix res = Whh.dot(ResetGate * prevState) + Whx.dot(input);
    return res.apply(tanh);
}

Matrix GRU::calcHiddenState(Matrix &UpdateGate, Matrix &prevState, Matrix &CandidateHiddenState) {
    Matrix res = (Matrix::ones(UpdateGate.rows,UpdateGate.cols) - UpdateGate) * CandidateHiddenState + UpdateGate * prevState;
    return res;
}

std::vector<Cache> GRU::feedForward(std::vector<Matrix> input) {
    std::vector<Cache> cache;
    Matrix prevState = Matrix(hiddenDims, 1);
    prevState.fill(0);
    for (int i = 0; i<input.size();i++){
        Matrix ResetGate = calcResetGate(input[i], prevState);
        Matrix UpdateGate = calcUpdateGate(input[i],prevState);
        Matrix CandidateHiddenState = calcCandidateHiddenState(input[i],prevState, ResetGate);
        Matrix HiddenState = calcHiddenState(UpdateGate, prevState, CandidateHiddenState);
        Cache newCache{ResetGate, UpdateGate, CandidateHiddenState, HiddenState};
        cache.push_back(newCache);
        prevState = HiddenState;
    }
    return cache;
}

void GRU::backProp(std::vector<Matrix> output, std::vector<Cache> cache, std::vector<Matrix> &input) {
    // calculate outGrad
        // for each cache, get the gradient dL/dy

    // for each layer, calculate the sum of the gradients for weight updates
    Matrix outGrad = Matrix::zeros(output[0].rows, output[0].cols);
    for (int i = 0;i<cache.size();i++){
        outGrad = output[i] - cache[i].HiddenState + outGrad;
    }

    Matrix WrxGradient = Matrix(this->Wrx.rows, this->Wrx.cols);
    Matrix WrhGradient = Matrix(this->Wrh.rows, this->Wrh.cols);
    Matrix WzxGradient = Matrix(this->Wzx.rows, this->Wzx.cols);
    Matrix WzhGradient = Matrix(this->Wzh.rows, this->Wzh.cols);
    Matrix WhxGradient = Matrix(this->Whx.rows, this->Whx.cols);
    Matrix WhhGradient = Matrix(this->Whh.rows, this->Whh.cols);
    for (int i = 0; i< cache.size();i++){
        // for each iteration, get all the gradients of the weights and sum them
        Matrix prevState = i == 0 ? Matrix::zeros(cache[0].HiddenState.rows, cache[0].HiddenState.cols) : cache[i-1].HiddenState;
        WrxGradient = WrhGradient + calcWrxGradient(cache[i], outGrad, prevState, input[i]);
        WrhGradient = WrhGradient + calcWrhGradient(cache[i], outGrad, prevState, input[i]);
        WzxGradient = WzxGradient + calcWzxGradient(cache[i], outGrad, prevState, input[i]);
        WzhGradient = WzhGradient + calcWzhGradient(cache[i], outGrad, prevState);
        WhxGradient = WhxGradient + calcWhxGradient(cache[i], outGrad, input[i]);
        WhhGradient = WhhGradient + calcWhhGradient(cache[i], outGrad, prevState);
    }

    // update all weights

    Wrx = Wrx - WrxGradient.scale(0.5);
    Wrh = Wrh - WrhGradient.scale(0.5);
    Wzx = Wzx - WzxGradient.scale(0.5);
    Wzh = Wzh - WzhGradient.scale(0.5);
    Whx = Whx - WhxGradient.scale(0.5);
    Whh = Whh - WhhGradient.scale(0.5);

}

Matrix GRU::calcWrxGradient(Cache &cache, Matrix &outGrad, Matrix &prevState, Matrix &input) {
    // dUr = xT . d16
    // d16 = d15 * (rt * (1-rt))
    // d15 = d10 * ht-1
    // d10 = d8 . WhhT
    // d8 = d6 * (1 - h~^2)
    // d6 = (1 - zt) * d0

    Matrix d6 = Matrix::ones(outGrad.rows, outGrad.cols) - cache.UpdateGate;
    Matrix d8 = d6 * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.UpdateGate * cache.UpdateGate);
    Matrix d10 = d8.dot(Whh);
    Matrix d15 = d10 * prevState;
    Matrix d16 = d15 * cache.ResetGate * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.ResetGate);
    return input.transpose().dot(d16);
}

Matrix GRU::calcWrhGradient(Cache &cache, Matrix &outGrad, Matrix &prevState, Matrix &input) {
    // dWr = h(t-1)T . d16
    // d16 = d15 * (rt * (1-rt))
    // d15 = d10 * ht-1
    // d10 = d8 . WhhT
    // d8 = d6 * (1 - h~^2)
    // d6 = (1 - zt) * d0

    Matrix d6 = Matrix::ones(outGrad.rows, outGrad.cols) - cache.UpdateGate;
    Matrix d8 = d6 * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.UpdateGate * cache.UpdateGate);
    Matrix d10 = d8.dot(Whh);
    Matrix d15 = d10 * prevState;
    Matrix d16 = d15 * cache.ResetGate * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.ResetGate);
    return prevState.transpose().dot(d16);
}

Matrix GRU::calcWzxGradient(Cache &cache, Matrix &outGrad, Matrix &prevState, Matrix &input) {
    // dUz = xT . d7
    // d7 = d5 * (zt * (1-zt))
    // d5 = d2 + d4
    // d4 = -1 * d3
    // d3 = h~ * d0
    // d2 = h(t-1) * d0
    Matrix d2 = prevState * outGrad;
    Matrix d3 = cache.CandidateHiddenState * outGrad;
    Matrix d5 = d2 - d3;
    Matrix d7 = d5 * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.UpdateGate) * cache.UpdateGate;
    return input.transpose().dot(d7);
}

Matrix GRU::calcWzhGradient(Cache &cache, Matrix &outGrad, Matrix &prevState) {
    // dWz = h(t-1)T . d7
    // d8 = d6 * (1-h~^2)
    // d6 = (1 - zt) * d0
    Matrix d2 = prevState * outGrad;
    Matrix d3 = cache.CandidateHiddenState * outGrad;
    Matrix d5 = d2 - d3;
    Matrix d7 = d5 * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.UpdateGate) * cache.UpdateGate;
    return prevState.transpose().dot(d7);
}

Matrix GRU::calcWhxGradient(Cache &cache, Matrix &outGrad, Matrix &input){
    // dUh = xT . d8
    Matrix d6 = outGrad * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.UpdateGate);
    Matrix d8 = d6 * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.CandidateHiddenState * cache.CandidateHiddenState);
    return input.transpose().dot(d8);
}

Matrix GRU::calcWhhGradient(Cache &cache, Matrix &outGrad, Matrix &prevState){
    Matrix d6 = outGrad * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.UpdateGate);
    Matrix d8 = d6 * (Matrix::ones(outGrad.rows, outGrad.cols) - cache.CandidateHiddenState * cache.CandidateHiddenState);
    return (prevState * cache.ResetGate).transpose().dot(d8);
}

Matrix GRU::getResult(std::vector<Cache> cache) {
    return cache.back().HiddenState;
}

double GRU::activation(double x) {
    return 1/(1+exp(-x));
}
