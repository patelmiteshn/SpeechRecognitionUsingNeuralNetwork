#include <iostream>
#include <exception>
#include <string>
#include <boost/scoped_ptr.hpp> // boost::scoped_ptr
using namespace boost;
using namespace std;
#include "neuralnetwork.h"



int main(int argc, char** argv)
{
    int nret = 1;
    try {
         boost::scoped_ptr<NeuralNetwork> neuralNetwork(new NeuralNetwork());
        if (argc < 2) {
            neuralNetwork->exit_with_help();
        }else { // if we got enough parameters...
            neuralNetwork->parse_command_line(argc,argv);
            if(neuralNetwork->trainTestFlag_ == 0)
                neuralNetwork->trainValidateNeuralNetwork();
            if(neuralNetwork->trainTestFlag_ == 1){
                neuralNetwork->loadTrainedModel();
                neuralNetwork->testNeuralNetwork();
            }
        }
    }catch(const std::exception& e) {
        nret = 0;
    }

    return nret;
}
