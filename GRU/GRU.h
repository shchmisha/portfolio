//
// Created by Миша Щербаков on 21.07.2022.
//

#ifndef NEURALNETWORK_GRU_H
#define NEURALNETWORK_GRU_H

#include "vector"
#include "cmath"
#include "../Matrix/Matrix.h"
#include "memory"

//class BaseNeuron{
//private:
//    std::vector<double> outputWeights;
//    int index;
//    double state, inputSum, gradient;
//    static double randomWeight();
//    static double activation(double x);
//    static double activationDerivative(double x);
//public:
//    BaseNeuron(int numOutputs, int index);
//    void hiddenBackprop(const std::vector<BaseNeuron> &nextLayer);
//    void outputBackprop(double targetVal);
//    void feedForward(const std::vector<BaseNeuron> &prevLayer);
//    void updateInputWeights();
//    double getState();
//    void setState(double state);
//};

struct Cache{
    Matrix ResetGate;
    Matrix UpdateGate;
    Matrix CandidateHiddenState;
    Matrix HiddenState;
};

class GRU{
private:
    int inputDims, hiddenDims, outputDims;
    Matrix Wrx, Wrh, Wzx, Wzh, Whx, Whh;
    static double activation(double x);
    Matrix calcResetGate(Matrix &input, Matrix &prevState);
    Matrix calcUpdateGate(Matrix &input, Matrix &prevState);
    Matrix calcCandidateHiddenState(Matrix &input, Matrix &prevState, Matrix &ResetGate);
    Matrix calcHiddenState(Matrix &UpdateGate, Matrix &prevState, Matrix &CandidateHiddenState);
    Matrix calcWrxGradient(Cache &cache, Matrix &outGrad, Matrix &prevState, Matrix &input);
    Matrix calcWrhGradient(Cache &cache, Matrix &outGrad, Matrix &prevState, Matrix &input);
    Matrix calcWzxGradient(Cache &cache, Matrix &outGrad, Matrix &prevState, Matrix &input);
    Matrix calcWzhGradient(Cache &cache, Matrix &outGrad, Matrix &prevState);
    Matrix calcWhxGradient(Cache &cache, Matrix &outGrad, Matrix &input);
    Matrix calcWhhGradient(Cache &cache, Matrix &outGrad, Matrix &prevState);

public:
    GRU(int input, int hidden, int output);
    std::vector<Cache> feedForward(std::vector<Matrix> input);
    void backProp(std::vector<Matrix> output, std::vector<Cache> cache, std::vector<Matrix> &input);
    Matrix getResult(std::vector<Cache> cache);
};

#endif //NEURALNETWORK_GRU_H


