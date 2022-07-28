//
// Created by Миша Щербаков on 26.07.2022.
//

#ifndef NEURALNETWORK_NN_H
#define NEURALNETWORK_NN_H

#include "vector"
#include "iostream"
#include "cstdlib"
#include "cmath"

class Neuron {
private:
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void);
    std::vector<double> m_outputWeights;
    unsigned m_neuronIndex;
    double m_outputVal;
    double m_inputVal;
    double m_gradient;
public:
    Neuron(unsigned numOutputs, unsigned my_index);
    void setOutputVal(double outputVal);
    double getOutputVal() const;
    void calcHiddenGradients(const std::vector<Neuron> &nextLayer);
    void calcOutputGradients(double targetVal);
    void feedForward(const std::vector<Neuron> &prevLayer);
    void updateInputWeights(std::vector<Neuron> &prevLayer);
};

class Net{
private:
    std::vector<std::vector<Neuron>> m_layers;
    double m_error;
public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
};


#endif //NEURALNETWORK_NN_H
