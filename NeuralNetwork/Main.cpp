#include "Network.h"
#include "TrainingData.h"

int main() 
{
	TrainingData trainData("trainingData.txt");

	std::vector<unsigned> topology;
	trainData.getTopology(topology);
	Network network(topology);

	std::vector<double> input_values, target_values, result_values;
	uint32_t trainingPass = 0;

	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << std::endl << "Pass " << trainingPass;

		if (trainData.getNextInputs(input_values) != topology[0]) 
		{
			printf("Error: Incorrectly quantity of input values based on Topology. Topology is %d", topology[0]);
			break;
		}

		// feed forward, calculate outputs of each neuron in next layer
		trainData.PrintData(": Inputs:", input_values);
		network.feedForward(input_values);

		// get the results of the final layer output
		network.getResults(result_values);
		trainData.PrintData("Outputs:", result_values);

		// get the target outputs
		trainData.getTargetOutputs(target_values);
		trainData.PrintData("Targets:", target_values);

		assert(target_values.size() == topology.back());

		// perform back propagation depending on the maginitude of the overall net error
		network.backProp(target_values);

		std::cout << "Net recent average error: " << network.getRecentAverageError() << std::endl;
	}

	std::cout << "Done" << std::endl;

	std::cin.get();
	return 0;
}