#pragma once

/*
	C++ Class Includes
*/
#include <iostream>
#include <assert.h>

// My Class Includes
#include "Neuron.h"

/*
	The Network class handles the forward propagation, generating the outputs for neurons and performing a
	back prop depending on the resulting overall net error
*/
class Network
{
public:
	/* Constructer */
	Network(const std::vector<unsigned> & topology);

	/* Feed Forward. got through each neuron and sum up weights and input values to get output value */
	void feedForward(const std::vector<double> & input_values);

	/* perform a back prop depending on how wrong the final output is (based on the overall network error value) */
	void backProp(const std::vector<double> & target_values);

	/* get the final results from the neural network and store the values in a vectorlist */
	void getResults(std::vector<double>& result_values) const;

	/* Getter. get the most recent net error value(target output - actually output) */
	double getRecentAverageError() const { return m_RecentAverageError; }
private:
	std::vector<Layer> m_Layers; // list of layers (a layer being just a vector list of neurons)
	double m_Error; // net error value
	double m_RecentAverageError; // most net error value
	double m_RecentAverageSmoothingFactor; // smoothing factor for the net error value
};

