//
// Created by Миша Щербаков on 26.07.2022.
//

#include "NN.h"

double Neuron::randomWeight(void) {
    return rand() / double(RAND_MAX);
}

Neuron::Neuron(unsigned int numOutputs, unsigned my_index) {

    for (unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(randomWeight());
    }
    this->m_neuronIndex = my_index;

}

void Neuron::setOutputVal(double outputVal) {
    m_outputVal = outputVal;
}

double Neuron::getOutputVal() const {
    return m_outputVal;
}

double Neuron::transferFunction(double x) {
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    return 1.0 - tanh(x) * tan(x);
}

void Neuron::calcHiddenGradients(const std::vector<Neuron> &nextLayer) {
    double sum = 0.0;
    for (int i =0; i<nextLayer.size();i++){
        sum+=nextLayer[i].m_outputVal * transferFunctionDerivative(nextLayer[i].m_inputVal) * nextLayer[i].m_outputVal;
    }
    m_gradient = sum;
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta;
}

void Neuron::feedForward(const std::vector<Neuron> &prevLayer) {
    double sum = 0.0;

    // sum the previous layer's outputs
    for (unsigned n=0;n<prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_neuronIndex];
    }
    m_inputVal = sum;
    m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::updateInputWeights(std::vector<Neuron> &prevLayer) {
    for (unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];

        double deltaWeight = m_gradient * transferFunctionDerivative(m_inputVal) * prevLayer[n].m_outputVal;

        neuron.m_outputWeights[m_neuronIndex] = 0.5 * deltaWeight;
    }
}

// ******************************************************************************************

Net::Net(const std::vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum<numLayers; ++layerNum){
        m_layers.push_back(std::vector<Neuron>());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
//        std::cout << numOutputs << std::endl;
        for (unsigned neuronNum = 0; neuronNum<topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Made a neuron in layer " << layerNum+1 <<std::endl;
        }
    }
}

void Net::feedForward(const std::vector<double> &inputVals) {
    // set outputs for the input layer of neurons
    assert(inputVals.size() == m_layers[0].size()-1);
    for (unsigned i=0; i<m_layers.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagate
    for (unsigned layerNum = 1; layerNum<m_layers.size(); ++layerNum){
        std::vector<Neuron> &prevLayer = m_layers[layerNum-1];
        for (unsigned n = 0; n<m_layers[layerNum].size(); ++n){
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals) {

    // calculate overall net error (RMS of output neuron errors)
    // RMS = sqrt(1/n * sum(target - actual) ^2)

    std::vector<Neuron> &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n=0; n<outputLayer.size() -1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error);

    // calculate output layer gradients

    for (unsigned n = 0; n<outputLayer.size(); ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // calculate gradients on hidden layers

    for (unsigned layerNum = m_layers.size()-2; layerNum>0; --layerNum){
        std::vector<Neuron> &hiddenLayer = m_layers[layerNum];
        std::vector<Neuron> &nextLayer = m_layers[layerNum+1];

        for (unsigned n = 0; n<hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // for all layers from outputs to first hidden layer, update connection weights

    for (unsigned layerNum = m_layers.size()-1; layerNum>0; --layerNum){
        std::vector<Neuron> &layer = m_layers[layerNum];
        std::vector<Neuron> &prevLayer = m_layers[layerNum];

        for (unsigned n = 0; n<layer.size(); ++n){
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &resultVals) const {
    resultVals.clear();

    for (unsigned n = 0; n<m_layers.back().size()-1;++n){
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}