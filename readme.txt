EXAMPLE TO TRAIN AND TEST NEURAL NETWORK 
created by Mitesh Patel on <March 2007>
amended by Mitesh Patel on <Feb 2014>

PRE-REQUISITE
 • boost/ublas libraries 1.5 or greater (older libraries should work but haven’t tested)
 • gcc4.3 compiler 

CURRENT NEURAL NETWORK (NN) ARCHITECTURE
 • Current code can only handle one hidden layer
 • Current setup required the data be separated training-validation-testing before hand
 • The NN architecture is limited to a feed forward neural network with back propagation.
 • The NN uses bipolar logistic function as the activation function.

  INPUT NODES:
  ⁃ The number of input nodes of the NN is calculated from the input of the training data file. 
  ⁃ It used the length of the rows in the data file to determine the number of input nodes
  HIDDEN NODES:
  ⁃ The default number of hidden nodes in the single layer are calculated using the formula as under:
  ⁃ (hiddenNodes_ = ceil((pow(outputNodes_,2.0) + outputNodes_+ 2)/2)+1. 
  ⁃ The number of hidden nodes can also be specified through the command line. 
  OUTPUT NODES: 
  ⁃ The number of output nodes correspond to the number of classes to be predicted.

  DATA FORMAT:
    TRAINING NEURAL NETWORK
    ⁃ When training the NN, the program expects five types of files. 
    ⁃ Training data file
    ⁃ training label file 
    ⁃ validation data file
    ⁃ validation label file
    ⁃ the name of the model file in which the trained parameters of the NN will be saved.

      TRAINING DATA FILE:
      ⁃ Each row of the training data file is a sample for training and the columns are the features.
      TRAINING LABEL FILE:
      ⁃ Each row in the label file represents the label of the corresponding sample in the training data file.
      VALIDATION DATA FILE:
      ⁃ Each row in the label file represents the label of the corresponding sample in the data file.
      VALIDATION LABEL FILE:
      ⁃ Each row in the label file represents the label of the corresponding sample in the validation data file.
      MODEL FILE:
      ⁃ The model file specified is used to store the trained parameters of the Neural Network.

    TESTING NEURAL NETWORK
    ⁃ When testing the NN, the program expects three types of files. 
    ⁃ Testing data file
    ⁃ Testing label file
    ⁃ the name of the model file from where the NN parameters are loaded.
      TESTING DATA FILE:
      ⁃ Each row in the label file represents the label of the corresponding sample in the data file.
      TESTING LABEL FILE:
      ⁃ Each row in the label file represents the label of the corresponding sample in the testing data file.
      MODEL FILE:
      ⁃ The trained parameters of the NN are loaded from this file.

COMMAND LINE ARGUMENTS
    FOR TRAINING
    ⁃ ./NeuralNetwork -t train [options] training_data.txt training_label.txt validation_data.txt validation_label.txt trained_model.txt
    ⁃ [options]
    ⁃ "-h number of hidden_nodes : (default calculated using (hiddenNodes_ = ceil((pow(outputNodes_,2.0) + outputNodes_+ 2)/2)+1 \n"
    ⁃ "-c training cycles : iteration for optimising the weights of NN (default 300)\n"
    ⁃“"-v displays NN parameters : displays the trained parameters of the model (default will not display)\n"
    FOR TESTING
    ⁃ ./NeuralNetwork -t test testing_data.txt testing_label.txt trained_model.txt




