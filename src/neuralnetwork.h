#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>       // used for <vector>
#include <exception>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <ctime>
#include <cmath>

// Boost
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/numeric/conversion/converter_policies.hpp>
#include <boost/random.hpp>

using namespace std;
using namespace boost::numeric::ublas;
using namespace boost;
#define MAXBUFSIZE 500000
#define NUMBEROFTRAININGCYCLE 300
#define LEARNINGCONSTANT 0.01

#define NEURAL_NETWORK_TRAINING_UPDATE_DEBUG_INFO
#define NEURAL_NETWORK_PARAMETER_DEBUG_INFO
//#define NEURAL_NETWORK_TRAINING_DEBUG_INFO
//#define NEURAL_NETWORK_VALIDATION_DEBUG_INFO
//#define NEURAL_NETWORK_TESTING_DEBUG_INFO
//#define RANDOM_WEIGHT_GENERATOR_DEBUG_INFO
//#define DATA_LOADING_DEBUG_INFO
//#define MODEL_PARAMETER_LOADING_DEBUG_INFO
//#define COMMANDLINE_ARGUMENT_PARSING_DEBUG_INFO


class NeuralNetwork
{

    matrix<double> loadDataSet(char* fileName);
    void validateNeuralNetwork();
    void trainNeuralNetwork();
    void saveTrainedModel();
    double randomNumberGenerator();

    //boost matrices used for various mathematical operation
    matrix<double> trainingData_;
    matrix<double> trainingLabels_;
    matrix<double> validationData_;
    matrix<double> validationLabels_;
    matrix<double> testingData_;
    matrix<double> testingLabels_;
    matrix<double> eValidation_;
    matrix<double> cyclicError_;
    matrix<double> nnWeight_;
    matrix<double> nnWeightBar_;
    matrix<double> wBest_;
    matrix<double> wbarBest_;

    //Global variables
    int inputNodes_;
    int outputNodes_;
    int bestIndex_;
    double lowestError_;
    int step_;
    double learnRate_;
    int cycle_;
    int numCycle_;
    int hiddenNodes_;
    bool verbose_;
    int predictionCount_;
    bool hiddenNodeDefaultFlag_;
    bool printInfoFlag_;

    //Pointers for file names to be loaded/saved
    char* trainingDataFile_;
    char* trainingDataFileLabel_;
    char* validationDataFile_;
    char* validationDataFileLabel_;
    char* modelFile_;
    char* testingDataFile_;
    char* testingDataFileLabel_;


public:
    NeuralNetwork();
    void trainValidateNeuralNetwork();
    void testNeuralNetwork();
    void exit_with_help();
    void parse_command_line(int argc, char **argv);
    void loadTrainedModel();
    bool trainTestFlag_;


};

#endif // NEURALNETWORK_H
