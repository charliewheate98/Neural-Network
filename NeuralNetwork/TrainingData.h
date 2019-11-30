#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

class TrainingData
{
public:
	TrainingData(const std::string filename);
	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(std::vector<unsigned>& topology);
	void PrintData(std::string label, std::vector<double>& v);

	unsigned getNextInputs(std::vector<double>& inputVals);
	unsigned getTargetOutputs(std::vector<double>& targetOutputVals);
private:
	std::ifstream m_trainingDataFile;
};

