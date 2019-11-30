#include "Network.h"

Network::Network(const std::vector<unsigned>& topology)
{
	unsigned num_layers = topology.size();

	for (unsigned layerNum = 0; layerNum < num_layers; ++layerNum) 
	{
		m_Layers.emplace_back(Layer());
		unsigned num_outputs_neurons = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_Layers.back().emplace_back(Neuron(num_outputs_neurons, neuronNum));
		}

		m_Layers.back().back().SetOutputValue(1.0);
	}
}

void Network::feedForward(const std::vector<double>& input_values)
{
	assert(input_values.size() == m_Layers[0].size() - 1);

	for (auto i = 0; i < input_values.size(); ++i) 
	{
		m_Layers[0][i].SetOutputValue(input_values[i]);
	}

	for (unsigned layerNum = 1; layerNum < m_Layers.size(); ++layerNum) 
	{
		Layer & previous_layer = m_Layers[layerNum - 1];

		for (unsigned n = 0; n < m_Layers[layerNum].size() - 1; ++n) 
		{
			m_Layers[layerNum][n].feedForward(previous_layer);
		}
	}
}

void Network::backProp(const std::vector<double> & target_values)
{
	/* Calculate overall net error */
	Layer & output_layer = m_Layers.back();
	
	m_Error = 0.0;

	for (auto n = 0; n < output_layer.size() - 1; ++n)
	{
		double delta = target_values[n] - output_layer[n].GetOutputValue();
		m_Error += delta * delta;
	}
	m_Error /= output_layer.size() - 1;
	m_Error = sqrt(m_Error);

	// implement a recent average measurement for DEBUG purposes
	m_RecentAverageError = (m_RecentAverageError * m_RecentAverageSmoothingFactor + m_Error) / (m_RecentAverageSmoothingFactor + 1.0);

	// calculate the output layer error gradients
	for (auto n = 0; n < output_layer.size() - 1; ++n)
	{
		output_layer[n].calcOutputGradients(target_values[n]);
	}

	// calculate the hidden layer error gradients
	for (unsigned layerNum = m_Layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer& hiddenLayer = m_Layers[layerNum];
		Layer& nextLayer = m_Layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// for all layers from outputs  to first hidden layer, 
	// update connection weights
	for (unsigned layerNum = m_Layers.size() - 1; layerNum > 0; --layerNum) {
		Layer& layer = m_Layers[layerNum];
		Layer& prevLayer = m_Layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Network::getResults(std::vector<double> & result_values) const
{
	result_values.clear();

	for (auto n = 0; n < m_Layers.back().size() - 1; ++n)
	{
		result_values.emplace_back(m_Layers.back()[n].GetOutputValue());
	}
}
