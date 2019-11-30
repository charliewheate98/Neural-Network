#include "TrainingData.h"

TrainingData::TrainingData(const std::string filename)
{
	m_trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(std::vector<unsigned>& topology)
{
	std::string line;
	std::string label;

	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;

	if (this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

unsigned TrainingData::getNextInputs(std::vector<double>& inputVals)
{
	inputVals.clear();

	std::string line;
	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.emplace_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double>& targetOutputVals)
{
	targetOutputVals.clear();

	std::string line;
	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}

void TrainingData::PrintData(std::string label, std::vector<double>& v)
{
	std::cout << label << " ";

	for (unsigned i = 0; i < v.size(); ++i) {
		std::cout << v[i] << " ";
	}

	std::cout << std::endl;
}
