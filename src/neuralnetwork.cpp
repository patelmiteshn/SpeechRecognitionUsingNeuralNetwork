/* ***************************************************************************************
 * The setup trains the Neural Network (NN) on the set of files provided for training
 * and validation and saves the trained model.
 * NEURAL NETWORK: The network architecture is a feed forward neural network with back propagation
 * and has a single hidden layer. The NN used bipolar logistic function as the activation function.
 * INPUT NODES: The number of input nodes of the NN is calculated from the input of the training data
 * file. It used the length of the rows in the data file to determine the number of input nodes
 * HIDDEN NODES: The default number of hidden nodes in the single layer are calculated using the
 * forumlae (hiddenNodes_ = ceil((pow(outputNodes_,2.0) + outputNodes_+ 2)/2)+1. The number of hidden
 * nodes can also be specified through the command line.
 * OUPUT NODES: The number of output nodes correspond to the number of classes to be predicted.
 *
 * DATA FORMAT:
 * TRAINING NEURAL NETWORK
 * When training the NN, the program expects five types of files. Training data file, training label file
 * validation data file, validation label file and the name of the model file in which the trained parameters
 * of the NN will be saved.
 * TRAINING DATA FILE: Each row of the training data file is a sample for training and the columns are the
 * features
 * TRAINING LABEL FILE: Each row in the label file represents the label of the corresponding sample in the data file.
 * VALIDATION DATA FILE: Each row of the validation data file is a sample for validating the training
 * weights and the columns are the features
 * VALIDATION LABEL FILE: Each row in the label file represents the label of the corresponding sample in the validation data file.
 *
 * TESTING NEURAL NETWORK
 * When testing the model, the program expects three types of files. Testing data file, testing label file
 * and the name of the model file from where the trained parameters of the NN are loaded.
 * TESTING DATA FILE: Each row of the training data file is a sample for training and the columns are the
 * features
 * TESTING LABEL FILE: Each row in the label file represents the label of the coresponding sample in the data file.
 * MODEL FILE: The trained parameters of the NN are loaded from this file
 *
 * created by Mitesh Patel on <March 2007>
 * amended by Mitesh Patel on <Feb 2014>
*/
#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork()
{
    bestIndex_ = 1;
    lowestError_ = 100;
    hiddenNodes_ = 0;
    inputNodes_ = 0;
    outputNodes_ = 0;
    numCycle_ = NUMBEROFTRAININGCYCLE;
    cycle_ = 0;
    step_ = 0;
    learnRate_ = LEARNINGCONSTANT;
    trainTestFlag_ = 0;
    predictionCount_ = 0;
    hiddenNodeDefaultFlag_ = 0;
    verbose_ = 0;
}

void NeuralNetwork::trainValidateNeuralNetwork(){

    // Loading training and validation data and corresponding label files
    trainingData_ = loadDataSet(trainingDataFile_);
    trainingData_ = trans(trainingData_);
    trainingLabels_ = loadDataSet(trainingDataFileLabel_);
    trainingLabels_ = trans(trainingLabels_);
    validationData_ = loadDataSet(validationDataFile_);
    validationData_ = trans(validationData_);
    validationLabels_ = loadDataSet(validationDataFileLabel_);
    validationLabels_ = trans(validationLabels_);

    #ifdef NEURAL_NETWORK_PARAMETER_DEBUG_INFO
        cout << "matrix training data size: " << trainingData_.size1() << " rows and " << trainingData_.size2() << " columns" << endl;
        cout << "matrix training labels size: " << trainingLabels_.size1() << " rows and " << trainingLabels_.size2() << " columns" << endl;
        cout << "matrix validation data size: " << validationData_.size1() << " rows and " << validationData_.size2() << " columns" << endl;
        cout << "matrix validation label size: " << validationLabels_.size1() << " rows and " << validationLabels_.size2() << " columns" << endl;
    #endif

    //Defining Neural Network Parameters i.e. number of input nodes, hidden nodes and output nodes based on the training data
    inputNodes_ = (trainingData_.size1())+1;
    outputNodes_ = trainingLabels_.size1();
    if(hiddenNodeDefaultFlag_ == 0){
        double temp = (pow(outputNodes_,2.0) + outputNodes_+ 2)/2;
        hiddenNodes_ = ceil(log2(temp)) + 1;
    }
    #ifdef NEURAL_NETWORK_PARAMETER_DEBUG_INFO
        cout << "Neural Network input nodes: " << inputNodes_ << " hidden nodes: " << hiddenNodes_ << " output nodes: " << outputNodes_ << endl;
    #endif

    nnWeight_ = zero_matrix<double>(outputNodes_,hiddenNodes_);
    nnWeightBar_ = zero_matrix<double>(hiddenNodes_-1,inputNodes_);
    eValidation_ = zero_matrix<double>(1,numCycle_);
    cyclicError_ = zero_matrix<double>(1,numCycle_);

    //random weights generator
    for(size_t i = 0; i <  hiddenNodes_-1; i++){
        for(size_t j = 0; j <  inputNodes_; j++){
            nnWeightBar_(i,j) = 2*(randomNumberGenerator() - 0.5);
        }
    }

    for(size_t i = 0; i < outputNodes_ ; i++){
        for(size_t j = 0; j < hiddenNodes_ ; j++){
            nnWeight_(i,j) = 2*(randomNumberGenerator() - 0.5);
        }
    }

    #ifdef NEURAL_NETWORK_PARAMETER_DEBUG_INFO
        cout << "size of weights matrix from input nodes to hidden nodes: " << nnWeightBar_.size1() << " rows and " << nnWeightBar_.size2() << " columns" << endl;
        cout << "size of weights matrix from hidden nodes to output nodes: " << nnWeight_.size1() << " rows and " << nnWeight_.size2() << " columns" << endl;
    #endif

    for(size_t c = 0; c < numCycle_; c++){
        validateNeuralNetwork(); // validate the neural network with validation data
        trainNeuralNetwork();    // train the neural network with training data
    }//for(size_t c = 0; c < numCycle_; c++)

    if(verbose_ == true){
        cout << "Trained Neural Network Information with one hidden layer" << endl;
        cout << "Input Nodes: " << inputNodes_ << endl;
        cout << "Output Nodes: " << outputNodes_ << endl;
        cout << "Hidden Nodes: " << hiddenNodes_ << endl;
        cout << "Learning Rate: " << learnRate_ << endl;
        cout << "Number of Interation Cycles: " << numCycle_ << endl;
        cout << "Neural network optimised at interation number: " << bestIndex_ << endl;
        cout << "Optimised Weights from input to hidden nodes: " << wbarBest_ << endl;
        cout << "Optimised Weights from hidden to output nodes: " << wBest_ << endl;
    }
    saveTrainedModel();
    //cout << "Neural network trained and model parameters saved in file named " << modelFile_ << endl;
}




void NeuralNetwork::validateNeuralNetwork(){

    //define vbar and preceptron matrices
    matrix<double> validationDataColumn(inputNodes_,1);
    matrix<double> validationLabelsColumn(outputNodes_,1);
    matrix<double> vbar(hiddenNodes_-1,1);
    matrix<double> v(outputNodes_,1);
    matrix<double> Y(hiddenNodes_,1);
    matrix<double> Z(outputNodes_,1);

    //Validation Cycle
    for(size_t a = 0; a < validationData_.size2(); a++){

        for(size_t temp = 0; temp < validationLabels_.size1(); temp++){
            validationLabelsColumn(temp,0) = validationLabels_(temp,a);
        }

        for(size_t temp = 0; temp < validationData_.size1(); temp++){
            validationDataColumn(temp,0) = validationData_(temp,a);
        }
        validationDataColumn((inputNodes_)-1,0) = -1;
        #ifdef NEURAL_NETWORK_VALIDATION_DEBUG_INFO
            cout << "validation data: " << validationDataColumn << endl;
            cout << "validation Label column: " << validationLabelsColumn << endl;
        #endif

        vbar = prod(nnWeightBar_,validationDataColumn);

        for(size_t temp = 0; temp < vbar.size1();temp++){
            Y(temp,0) = (1 - exp(-vbar(temp,0)))/(1+exp(-vbar(temp,0)));
        }

        #ifdef NEURAL_NETWORK_VALIDATION_DEBUG_INFO
            cout << " values before multiplying with the bipolar logistic function(BLF) at the hidden layer: " << vbar << endl;
            cout << "perceptron values at the hidden layer: " << Y << endl;
        #endif


        Y(Y.size1()-1,0) = -1;
        v = prod(nnWeight_,Y);

        for(size_t temp = 0; temp < v.size1();temp++){
            Z(temp,0) = (1 - exp(-v(temp,0)))/(1+exp(-v(temp,0)));
        }

        #ifdef NEURAL_NETWORK_VALIDATION_DEBUG_INFO
            cout << " values before multiplying with the bipolar logistic function(BLF) at the output layer: " << v << endl;
            cout << "perceptron values at the output layer: " << Z << endl;
        #endif

        //validation error
        double temp = 0;
        for(size_t i = 0; i < Z.size1();i++){
            temp = temp + 0.5*pow((validationLabelsColumn(i,0) - Z(i,0)),2);
        }

        #ifdef NEURAL_NETWORK_VALIDATION_DEBUG_INFO
            cout << "Validation Error " << temp << endl;
        #endif
        eValidation_(0,cycle_) = temp;

    }//for(size_t a = 0; a < validationData_.size1(); a++)

    //save best weights
    if(eValidation_(0,cycle_) < lowestError_){
        wBest_ = nnWeight_;
        wbarBest_ = nnWeightBar_;
        bestIndex_ = cycle_;
        lowestError_ = eValidation_(0,cycle_);
    }

    #ifdef NEURAL_NETWORK_TRAINING_UPDATE_DEBUG_INFO
        cout << "Interation cycle number: " << cycle_ << endl;
        cout << "Neural network optimised at interation number: " << bestIndex_ << endl;
    #endif

    #ifdef NEURAL_NETWORK_VALIDATION_DEBUG_INFO
        cout << "Optimised Weights from input to hidden nodes: " << wbarBest_ << endl;
        cout << "Optimised Weights from hidden to output nodes: " << wBest_ << endl;
    #endif
}

void NeuralNetwork::trainNeuralNetwork(){

    //vbar and preceptron and back propogation matrices.
    matrix<double> vbar(hiddenNodes_-1,1);
    matrix<double> v(outputNodes_,1);
    matrix<double> Y(hiddenNodes_,1);
    matrix<double> Z(outputNodes_,1);
    matrix<double> trainingDataColumn(inputNodes_,1);
    matrix<double> trainingLabelsColumn(outputNodes_,1);
    matrix<double> delta(outputNodes_,1);
    matrix<double> deltaBar(hiddenNodes_-1,1);
    matrix<double> deltaBar1(hiddenNodes_-1,1);
    matrix<double> deltaBar2(hiddenNodes_-1,1);
    matrix<double> sampleError(1,numCycle_*trainingData_.size2());

    for(size_t s = 0; s < trainingData_.size2(); s++){

        for(size_t temp = 0; temp < trainingLabels_.size1();temp++){
            trainingLabelsColumn(temp,0) = trainingLabels_(temp,s);
        }

        for(size_t temp = 0; temp < trainingData_.size1();temp++){
            trainingDataColumn(temp,0) = trainingData_(temp,s);
        }
        trainingDataColumn((inputNodes_)-1,0) = -1;

        #ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "training data: " << trainingDataColumn << endl;
            cout << "training Label column: " << trainingLabelsColumn << endl;
        #endif

        vbar = prod(nnWeightBar_,trainingDataColumn);

        for(size_t temp = 0; temp < vbar.size1();temp++){
            Y(temp,0) = (1 - exp(-vbar(temp,0)))/(1+exp(-vbar(temp,0)));
        }
        #ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "Values before multiplying with the bipolar logistic function(BLF) at the hidden layer: " << vbar << endl;
            cout << "perceptron values at the hidden layer: " << Y << endl;
        #endif

        Y(Y.size1()-1,0) = -1;
        v = prod(nnWeight_,Y);

        for(size_t temp = 0; temp < v.size1();temp++){
            Z(temp,0) = (1 - exp(-v(temp,0)))/(1+exp(-v(temp,0)));
        }
        #ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "Values before multiplying with the bipolar logistic function(BLF) at the output layer: " << v << endl;
            cout << "perceptron values at the output layer: " << Z << endl;
        #endif

        //calculate delta back propagation
        for(size_t temp = 0; temp < Z.size1();temp++){
            delta(temp,0) = (trainingLabelsColumn(temp,0) - Z(temp,0))*(0.5*(1-pow(Z(temp,0),2)));
        }

        for(size_t temp = 0; temp < Y.size1()-1;temp++){
            deltaBar1(temp,0) = 0.5*(1-pow(Y(temp,0),2));
        }

        for(size_t i = 0; i < hiddenNodes_-1;i++){
            double sum = 0;
            for(size_t j = 0; j < outputNodes_;j++){
                sum = sum + delta(j,0)*nnWeight_(j,i);
            }
            deltaBar2(i,0) = sum;
        }

        for(size_t i = 0; i < hiddenNodes_-1;i++){
            deltaBar(i,0) = deltaBar1(i,0)*deltaBar2(i,0);
        }

        #ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "Error between predicted output and actual output: " << delta << endl;
            cout << "Back Propogation error between output and hidden layer: " << deltaBar2 << endl;
            //cout << "Back Propogation error between hidden and input layer: " << deltaBar1 << endl;
            cout << "Total Back Propogation error at hidden nodes: " << deltaBar << endl;
        #endif

        //Update weights
        double tmp;
        matrix<double> dw = zero_matrix<double>(outputNodes_,hiddenNodes_);
        matrix<double> YTemp;
        YTemp = trans(Y);
        for(size_t k = 0; k < 1; k++) {
            for(size_t i = 0; i < outputNodes_; i++) {
                tmp = delta(i,k);
                for(size_t j = 0; j < hiddenNodes_; j++) {
                    dw(i,j) = dw(i,j) + tmp * YTemp(k,j);
                }
            }
        }
        nnWeight_ = nnWeight_ + learnRate_ *(dw);

        matrix<double> dwBar = zero_matrix<double>(hiddenNodes_-1,inputNodes_);
        matrix<double> YTempBar;
        YTempBar = trans(trainingDataColumn);
        for(size_t k = 0; k < 1; k++) {
            for(size_t i = 0; i < hiddenNodes_-1; i++) {
                tmp = deltaBar(i,k);
                for(size_t j = 0; j < inputNodes_; j++) {
                    dwBar(i,j) = dwBar(i,j) + tmp * YTempBar(k,j);
                }
            }
        }

        nnWeightBar_ = nnWeightBar_ + learnRate_*(dwBar);

        #ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "updated weights connecting perceptrons from input to hidden nodes: " << nnWeightBar_ << endl;
            cout << "updated weights connecting perceptrons from hidden to output nodes: " << nnWeight_ << endl;
        #endif

        //error for every sample
        double temp = 0;
        for(size_t i = 0; i < trainingLabelsColumn.size1();i++){
            temp = temp + 0.5*pow((trainingLabelsColumn(i,0) - Z(i,0)),2);
        }
        sampleError(0,step_) = temp;
        step_ = step_ + 1;
        #ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "training error between predicted and actual label: " << temp << endl;
        #endif

    }//for(size_t s = 0; s < trainingData_.size1(); s++)

    //training error
    for(size_t i = step_ - trainingData_.size1()-1; i < step_-1 ;i++){
        cyclicError_(0,cycle_) = cyclicError_(0,cycle_) + sampleError(0,i);
    }
    cycle_ = cycle_ + 1;
}

void NeuralNetwork::testNeuralNetwork(){

    testingData_ = loadDataSet(testingDataFile_);
    testingData_ = trans(testingData_);
    testingLabels_ = loadDataSet(testingDataFileLabel_);
    testingLabels_ = trans(testingLabels_);

    #ifdef NEURAL_NETWORK_TESTING_DEBUG_INFO
        cout << "Testing data size: " << testingData_.size1() << " " << testingData_.size2() << endl;
        cout << "Testing label size: " << testingLabels_.size1() << " " << testingLabels_.size2() << endl;
    #endif

    matrix<double> testingDataColumn(inputNodes_,1);
    matrix<double> testingLabelsColumn(outputNodes_,1);
    matrix<double> vbar(hiddenNodes_-1,1);
    matrix<double> v(outputNodes_,1);
    matrix<double> Y(hiddenNodes_,1);
    matrix<double> Z(outputNodes_,1);

    for(size_t a = 0; a < testingData_.size2(); a++){

        for(size_t temp = 0; temp < testingLabels_.size1(); temp++){
            testingLabelsColumn(temp,0) = testingLabels_(temp,a);
        }

        for(size_t temp = 0; temp < testingData_.size1(); temp++){
            testingDataColumn(temp,0) = testingData_(temp,a);
        }
        testingDataColumn((inputNodes_)-1,0) = -1;
        /*#ifdef NEURAL_NETWORK_TESTING_DEBUG_INFO
            cout << "Testing data: " << testingDataColumn << endl;
            cout << "testing label: " << testingLabelsColumn << endl;
        #endif*/

        vbar = prod(wbarBest_,testingDataColumn);

        for(size_t temp = 0; temp < vbar.size1();temp++){
            Y(temp,0) = (1 - exp(-vbar(temp,0)))/(1+exp(-vbar(temp,0)));
        }

        /*#ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "Values before multiplying with the bipolar logistic function(BLF) at the hidden layer: " << vbar << endl;
            cout << "perceptron values at the hidden layer: " << Y << endl;
        #endif*/

        Y(Y.size1()-1,0) = -1;
        v = prod(wBest_,Y);
        for(size_t temp = 0; temp < v.size1();temp++){
            Z(temp,0) = (1 - exp(-v(temp,0)))/(1+exp(-v(temp,0)));
        }

        /*#ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "Values before multiplying with the bipolar logistic function(BLF) at the hidden layer: " << v << endl;
            cout << "perceptron values at the hidden layer: " << Z << endl;
        #endif*/

        //extracting the actual label of the data from label file
        int maxVal = -100;
        int maxValIndex = 0;
        for(size_t temp = 0; temp < testingLabels_.size1(); temp++){
            testingLabelsColumn(temp,0) = testingLabels_(temp,a);
            if(testingLabelsColumn(temp,0) > maxVal){
                maxVal = testingLabelsColumn(temp,0);
                maxValIndex = temp+1;
            }

        }

        //extracting the predicted label of the data sample done by the neural network
        float maxVal1 = -100.00;
        int maxValIndexP = 0;
        for(size_t temp = 0; temp < Z.size1();temp++){
            if(Z(temp,0) > maxVal1){
                maxVal1 = Z(temp,0);
                maxValIndexP = temp+1;
            }
        }

        //calculating statistics
        if(maxValIndex == maxValIndexP)
            predictionCount_ +=1;

        #ifdef NEURAL_NETWORK_TRAINING_DEBUG_INFO
            cout << "actual value: " << maxValIndex << endl;
            cout << "predicted output: " << maxValIndexP << endl;
            cout << "predicted Count: " << predictionCount_ << endl;
        #endif
    }
    double calcAcc = testingData_.size2();
    cout << "Prediction Accuracy: " << (predictionCount_/calcAcc)*100 << endl;

}

matrix<double> NeuralNetwork::loadDataSet(char* fileName){

    //ifstream file;
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];
    ifstream infile;
    infile.open(fileName);
    if(!infile.is_open()){
        cout << "Failed to open file" << endl;
        exit(1);
    }

    while (! infile.eof()){
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);

        while(! stream.eof()){
            stream >> buff[cols*rows+temp_cols];
            //cout << buff[cols*rows+temp_cols] << " " << endl;
            temp_cols++;
        }

        if (temp_cols == 0)
            continue;
        if (cols == 0)
            cols = temp_cols;
        rows++;
    }
    infile.close();
    matrix<double> m(rows,cols);

    // Populate matrix with numbers.
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            m(i,j) = buff[ cols*i+j ];
            //cout << m(i,j) << " " << i << " " << j << endl;
        }
    }
    #ifdef DATA_LOADING_DEBUG_INFO
        cout << "matrix rows: " << rows << " and columns: " << cols << endl;
        cout << "populated matrix: " << m << endl;
    #endif
    return m;
}

double NeuralNetwork::randomNumberGenerator(){
      const double rangeMin = 0.0;
      const double rangeMax = 1.0;
      typedef boost::uniform_real<> NumberDistribution;
      typedef boost::mt19937 RandomNumberGenerator;
      typedef boost::variate_generator<RandomNumberGenerator&,
                                       NumberDistribution> Generator;

      NumberDistribution distribution(rangeMin, rangeMax);
      RandomNumberGenerator generator;
      Generator numberGenerator(generator, distribution);
      generator.seed(clock()); // seed with the current time

      #ifdef RANDOM_WEIGHT_GENERATOR_DEBUG_INFO
            cout << numberGenerator() << endl;
      #endif
      return numberGenerator();
}


void NeuralNetwork::saveTrainedModel(){
    FILE *fp = fopen(modelFile_,"w");
        if(fp==NULL){
            cout << "cannot write n the file" << endl;
            exit(1);
        }

        //fprintf(fp,"Feed forward neural network with back propogation\n");
        fprintf(fp,"input_Nodes %d\n", inputNodes_);
        fprintf(fp,"output_Nodes %d\n", outputNodes_);
        fprintf(fp,"hidden_Nodes %d\n", hiddenNodes_);
        fprintf(fp,"learning_Rate %f\n", learnRate_);
        fprintf(fp,"Training_Cycles %d\n", numCycle_);
        fprintf(fp,"Best_weights_at_interation_number %d\n", bestIndex_);

        //fprintf(fp,"trained model parameters for input to hidden nodes\n");
        fprintf(fp,"wbar\n");
        for(size_t i = 0; i <  hiddenNodes_-1; i++){
            for(size_t j = 0; j <  inputNodes_; j++){
                fprintf(fp, "%.16g ",wbarBest_(i,j));
            }
        }

        fprintf(fp, "\n");
        fprintf(fp,"w\n");
        //fprintf(fp,"trained model parameters from hidden nodes to output nodes\n");

        for(size_t i = 0; i < outputNodes_ ; i++){
            for(size_t j = 0; j < hiddenNodes_ ; j++){
                fprintf(fp, "%.16g ",wBest_(i,j));
            }
        }
        fprintf(fp, "\n");

        if (ferror(fp) != 0 || fclose(fp) != 0){
            cout << "error in writing the trained neural network parameters to the file" << endl;
            exit(1);
        }
        else
            cout << "Neural Network trained parameters saved in file named: " << modelFile_ << endl;
}


void NeuralNetwork::loadTrainedModel(){
    FILE *fp = fopen(modelFile_,"rb");
    bool matrixSetupFlag = false;
    bool matrixReadBreakFlag = false;

    cout << "Neural Network parameters loaded from the model file: " << modelFile_ << endl;
    if(fp==NULL){
        cout << "model file cannot be loaded" << endl;
        exit(1);
    }

    char cmd[81];
    while(1)
    {
        fscanf(fp,"%80s",cmd);

        if(strcmp(cmd,"input_Nodes")==0){
            fscanf(fp,"%d",&inputNodes_);
            //cout << "input nodes " << inputNodes_ << endl;
        }
        else if(strcmp(cmd,"output_Nodes")==0){
            fscanf(fp,"%d",&outputNodes_);
            //cout << "output nodes " << outputNodes_ << endl;
        }
        else if(strcmp(cmd,"hidden_Nodes")==0){
            fscanf(fp,"%d",&hiddenNodes_);
            //cout << "hidden nodes " << hiddenNodes_ << endl;
        }
        else if(strcmp(cmd,"learning_Rate")==0){
            fscanf(fp,"%f",&learnRate_);
            //cout << "learning rate " << learnRate_ << endl;
        }
        else if(strcmp(cmd,"Training_Cycles")==0){
            fscanf(fp,"%d",&numCycle_);
            //cout << "number of cycles " << numCycle_ << endl;
        }
        else if(strcmp(cmd,"Best_weights_at_interation_number")==0){
            fscanf(fp,"%d",&bestIndex_);
            //cout << "bestIndex " << bestIndex_ << endl;
            matrixSetupFlag = true;
        }
        if(matrixSetupFlag == true){
            wBest_ = zero_matrix<double>(outputNodes_,hiddenNodes_);
            wbarBest_ = zero_matrix<double>(hiddenNodes_-1,inputNodes_);
            matrixSetupFlag = false;
        }
        if(strcmp(cmd,"wbar")==0){
            while(1){
                for(size_t i = 0; i <  hiddenNodes_-1; i++){
                    for(size_t j = 0; j <  inputNodes_; j++){
                        fscanf(fp,"%80s",cmd);
                        if(feof(fp) || (strcmp(cmd,"w") == 0)){
                            matrixReadBreakFlag = true;
                            break;
                        }
                        wbarBest_(i,j) = atof(cmd);
                    }
                    if(matrixReadBreakFlag)
                        break;
                }
                if(matrixReadBreakFlag)
                    break;
            }
        }
        if(strcmp(cmd,"w")==0){
            matrixReadBreakFlag = false;
            while(1){
                for(size_t i = 0; i < outputNodes_ ; i++){
                    for(size_t j = 0; j < hiddenNodes_ ; j++){
                        fscanf(fp,"%80s",cmd);
                        if(feof(fp)){
                            matrixReadBreakFlag = true;
                            break;
                        }
                        wBest_(i,j) = atof(cmd);
                    }
                    if(matrixReadBreakFlag)
                        break;
                }
                if(matrixReadBreakFlag)
                    break;
            }
        }
        if(feof(fp))
            break;
    }
    #ifdef MODEL_PARAMETER_LOADING_DEBUG_INFO
        cout << "Input Nodes: " << inputNodes_ << endl;
        cout << "Output Nodes: " << outputNodes_ << endl;
        cout << "Hidden Nodes: " << hiddenNodes_ << endl;
        cout << "Learning Rate: " << learnRate_ << endl;
        cout << "Number of Interation Cycles: " << numCycle_ << endl;
        cout << "Neural network optimised at interation number: " << bestIndex_ << endl;
        cout << "Optimised Weights from input to hidden nodes: " << wbarBest_ << endl;
        cout << "Optimised Weights from hidden to output nodes: " << wBest_ << endl;
    #endif
}

void NeuralNetwork::exit_with_help()
{
    if(trainTestFlag_ == 0){
        printf(
        "Usage: NeuralNetwork [options] trainingDataFile trainingLabelFile validationDataFile validationLabelFile modelFile \n"
        "options:\n"
        "-t [train]\n"
        "-l learning_Rate : (default 0.1)\n"
        "-h number of hidden_nodes : (default calculated using (hiddenNodes_ = ceil((pow(outputNodes_,2.0) + outputNodes_+ 2)/2)+1 \n"
        "-c training cycles : iteration for optimising the weights of NN (default 300)\n"
        "-v displays NN parameters : displays the trained paramerters of the model (default will not display)\n"
        );
    }if(trainTestFlag_ == 1){
        printf(
        "Usage: NeuralNetwork [options] testingDataFile testingLabelFile modelFile \n"
        "options:\n"
        "-t [test]\n"
        "-v displays NN parameters : displays the trained paramerters of the model (default will display)\n"
        );
    }
    exit(1);
}

void NeuralNetwork::parse_command_line(int argc, char **argv){
    int i;
    //default values
    learnRate_ = LEARNINGCONSTANT;
    numCycle_ = NUMBEROFTRAININGCYCLE;
    verbose_ = 0;

    for(i = 1; i < argc; i++){
        if(argv[i][0] != '-')
            break;
        if(++i>=argc)
            break;
        switch(argv[i-1][1])
        {
            case 't':
                if(strcmp(argv[i],"train")==0){
                    trainTestFlag_ = 0;
                    //cout << "train test flag " << trainTestFlag_ << endl;
                }
                if(strcmp(argv[i],"test")==0){
                    trainTestFlag_ = 1;
                    //cout << "train test flag " << trainTestFlag_ << endl;
                }
                break;
            case 'l':
                learnRate_ = atof(argv[i]);
                //cout << "learing rate " << atof(argv[i]) << endl;
                break;
            case 'h':
                hiddenNodeDefaultFlag_ = 1;
                hiddenNodes_ = atoi(argv[i]);
                //cout << "hiddenNodes " << atoi(argv[i]) << endl;
                break;
            case 'c':
                numCycle_ = atoi(argv[i]);
                 //cout <<  "training cycles " << atoi(argv[i]) << endl;
                break;
            case 'v':
                verbose_ = atoi(argv[i]);
                //cout << "verbose " << atoi(argv[i]);
                break;
        }

    }

    if(i+5 == argc && trainTestFlag_ == 0){
        trainingDataFile_ = argv[i];
        trainingDataFileLabel_ = argv[i+1];
        validationDataFile_ = argv[i+2];
        validationDataFileLabel_ = argv[i+3];
        modelFile_ = argv[i+4];
        #ifdef COMMANDLINE_ARGUMENT_PARSING_DEBUG_INFO
            cout << "training data file name: " << trainingDataFile_ << endl;
            cout << "training data label file name: " << trainingDataFileLabel_ << endl;
            cout << "validation data file name: " << validationDataFile_ << endl;
            cout << "validation data label file name: " << validationDataFileLabel_ << endl;
            cout << "Neural Network Trained Model will be saved with file name: " << modelFile_ << endl;
        #endif
    }else if(i+3 == argc && trainTestFlag_ == 1){
        testingDataFile_ = argv[i];
        testingDataFileLabel_ = argv[i+1];
        modelFile_ = argv[i+2];
        #ifdef COMMANDLINE_ARGUMENT_PARSING_DEBUG_INFO
            cout << "testing data file name: " << testingDataFile_ << endl;
            cout << "testing data label file name: " << testingDataFileLabel_ << endl;
            cout << "Neural Network Testing Model file name: " << modelFile_ << endl;
        #endif
    }else{
        cout << "ask for help" << endl;
        exit_with_help();
    }

}
