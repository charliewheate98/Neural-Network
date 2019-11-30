#pragma once

#include <vector>
#include <cstdlib>

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned num_output_neurons, unsigned neuron_index);
	~Neuron() {}

	void feedForward(const Layer & previous_layer);
	void SetOutputValue(double val) { m_OutputValue = val; }
	void calcOutputGradients(double target_value);
	void calcHiddenGradients(const Layer & next_layer);
	void updateInputWeights(Layer & previous_layer);

	inline double GetOutputValue() const { return m_OutputValue; }
private:
	static double randomWeight(void) { return rand() / double(RAND_MAX); }

	static double activationFunction(double x);
	static double activationDerivativeFunction(double x);
	double sumDOW(const Layer & next_layer); // weights of current layer neurons * error gradient of next layer neurons

	static double learning_rate;
	static double alpha;

	unsigned m_NeuronIndex;
	double m_OutputValue;
	double m_Gradient;
	std::vector<Connection> m_outputConnections;
};

