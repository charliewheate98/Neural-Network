#include "Neuron.h"
#include <iostream>

double Neuron::learning_rate = 0.15;
double Neuron::alpha = 0.5;	

Neuron::Neuron(unsigned num_output_neurons, unsigned neuron_index) 
{
	m_NeuronIndex = neuron_index;

	for (unsigned c = 0; c < num_output_neurons; ++c)
	{
		m_outputConnections.emplace_back(Connection());
		m_outputConnections.back().weight = randomWeight();
	}
}

void Neuron::feedForward(const Layer& previous_layer)
{
	double sum = 0.0;

	for (auto n = 0; n < previous_layer.size(); ++n)
	{
		sum += previous_layer[n].GetOutputValue() * previous_layer[n].m_outputConnections[m_NeuronIndex].weight;
	}

	// pass through the activation function to normalize between [-1 .. 1]
	m_OutputValue = Neuron::activationFunction(sum);
}

void Neuron::calcOutputGradients(double target_value)
{
	double error = target_value - m_OutputValue;
	m_Gradient = error * Neuron::activationDerivativeFunction(m_OutputValue);
}

void Neuron::calcHiddenGradients(const Layer & next_layer)
{
	double dow = sumDOW(next_layer);
	m_Gradient = dow * Neuron::activationDerivativeFunction(m_OutputValue);
}

void Neuron::updateInputWeights(Layer& previous_layer)
{
	for (auto n = 0; n < previous_layer.size(); ++n)
	{
		Neuron & neuron = previous_layer[n];
		double oldDeltaWeight = neuron.m_outputConnections[m_NeuronIndex].deltaWeight;

		double newDeltaWeight =
			learning_rate
			* neuron.GetOutputValue()
			* m_Gradient
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputConnections[m_NeuronIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputConnections[m_NeuronIndex].weight += newDeltaWeight;
	}
}

double Neuron::activationFunction(double x)
{
	return tanh(x);
}

double Neuron::activationDerivativeFunction(double x)
{
	return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer& next_layer)
{
	double sum = 0.0;

	for (auto n = 0; n < next_layer.size() - 1; ++n)
	{
		sum += m_outputConnections[n].weight * next_layer[n].m_Gradient;
	}

	return sum;
}
