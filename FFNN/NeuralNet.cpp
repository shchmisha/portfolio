//
// Created by Миша Щербаков on 08.07.2022.
//

#include "NeuralNet.h"

double Neuron::randomWeight(void) {
    return rand() / double(RAND_MAX);
}

Neuron::Neuron(unsigned int numOutputs, unsigned my_index) {
    eta = 1.5;
    alpha = 0.5;
    for (unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    this->m_MyIndex = my_index;
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
    return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;

    // sum our contributions of the errors at the nodes we feed

    for (unsigned n = 0; n < nextLayer.size()-1; +n){
        sum+=m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;

    // sum the previous layer's outputs
    for (unsigned n=0;n<prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_MyIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_MyIndex].deltaWeight;

        double newDeltaWeight =
        // individual input, magnified by the gradient and train rate
            eta
            * neuron.getOutputVal()
            * m_gradient
        // also add momentum = a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;

        neuron.m_outputWeights[m_MyIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_MyIndex].weight += newDeltaWeight;
    }
}

// ************************** Network definitions **********************

Net::Net(const std::vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum<numLayers; ++layerNum){
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
//        std::cout << numOutputs << std::endl;
        for (unsigned neuronNum = 0; neuronNum<=topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Made a neuron in layer " << layerNum+1 <<std::endl;
        }

        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const std::vector<double> &inputVals) {
    // set outputs for the input layer of neurons
    assert(inputVals.size() == m_layers[0].size()-1);
    for (unsigned i=0; i<m_layers.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagate
    for (unsigned layerNum = 1; layerNum<m_layers.size()-1; ++layerNum){
        Layer &prevLayer = m_layers[layerNum-1];
        for (unsigned n = 0; n<m_layers[layerNum].size(); ++n){
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals) {

    // calculate overall net error (RMS of output neuron errors)
    // RMS = sqrt(1/n * sum(target - actual) ^2)

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n=0; n<outputLayer.size() -1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error);

    // implement a recent average measurement

    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // calculate output layer gradients

    for (unsigned n = 0; n<outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // calculate gradients on hidden layers

    for (unsigned layerNum = m_layers.size()-2; layerNum>0; --layerNum){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum+1];

        for (unsigned n = 0; n<hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // for all layers from outputs to first hidden layer, update connection weights

    for (unsigned layerNum = m_layers.size()-1; layerNum>0; --layerNum){
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum];

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