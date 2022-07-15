//
// Created by Миша Щербаков on 08.07.2022.
//

#ifndef BLOCKCHAIN_NEURALNET_H
#define BLOCKCHAIN_NEURALNET_H

#include "vector"
#include "iostream"
#include "cstdlib"
#include "cmath"

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
private:
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void);
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_MyIndex;
    double m_gradient;
    double eta;
    double alpha;
public:
    Neuron(unsigned numOutputs, unsigned my_index);
    void setOutputVal(double outputVal);
    double getOutputVal() const;
    void calcHiddenGradients(const Layer &nextLayer);
    void calcOutputGradients(double targetVal);
    void feedForward(const Layer &prevLayer);
    void updateInputWeights(Layer &prevLayer);
};

class Net{
private:
    std::vector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
};


#endif //BLOCKCHAIN_NEURALNET_H
