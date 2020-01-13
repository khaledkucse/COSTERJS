# DAMCA Recommender
The repository is a python project that has a Bi directional LSTM based Encoder Decoder with Attention mechanism and Beam Search tool. Primary purpose is to reccomend the API method call along with arguments for Java.

## Get Started
To know about the content and how it works lets looks at the files and folder we have in the repository.
Next we will learn how to install the project and use it for training, testing and infering. If we want to describe the DAMCA Recommender in a single sentence:
the project takes code tokens as input sequence and returns a list the method call with arguments as suggestion.

### Contains
The project contains several python scripts by which the DAMCA Recommender tool execute.

* __main__.py: This python script initiates the tool. User needs to run the scripts to train, test or infer.
* config.py: The configuration script has all the global variables. User need to change the value of those variable before running the __main__.py scripts
* AttentionMechanism.py: This script contains the properties of the custom Attention Layer in the neural network
* Beam_Search.py: Beam Search(BS) algorithm is written in this script
* CreateDataset.py: The script is responsible to create dataset for the learning/testing processes.
* DataPreprocessing.py: It process the input data and help the CreateDataset.py script to process the input data.
* Evaluation.py: The scripts execute the evaluation mechanisms(Accuracy, Recall, BLUE Score, MRR) on results of DAMCA Recommender.
* KerasNNStructure.py: The scripts contains the code for different neural network structure designed using Keras library.
* KerasOperation.py: From __main__.py scripts, different methods of KerasOperation.py is called when "which_implementation" of config.py is defined as "keras" . And those methods executes all functionalities with help of other scripts.
* TFEstimatorNNStructure.py: The scripts contains the code for different neural network structure designed using Tennsorflow Estimator library.
* TFEstimatorOperation.py: When global variable, "which_implementation" is defined as "tfestimator" then training and testing method defined in this scripts are called by __main__.py
* TFEstimatorPostProcess.py: Data post processing scripts for TF Estimator library.

There are some folders that contains specific elements for the program.
* /damca_dataset: It contains the training and testing data
* /damca_evaluation: It contains the evaluation reports after testing
* /damca_results: It contains the raw results for each test case. Evaluation.py uses these data to get the evaluation metrics.
* /damca_vocabulary: It has two file: input.vocab(vocabulary for input sequence) and output.vocab(vocabulary for output sequences)
* /damca_training_checkpoints: training checkpoints are stored in this folder when code with tfestimator is executed.
* /damca_models: training model created by keras library is stored in the directory.

### Requirements:
Following packages need to be installed in order to run the program:
```
python 3.5 or more
pip3
```

### Installation Step:

1. Clone the repository using following command: 
    ```
    git clone -b recommender --single-branch https://github.com/khaledkucse/DeepAPIMethodCumArgRecc.git
    ```
2. Next, install the dependency. Run following command:
    ```
    pip3 install -r requirements.txt
    ```
    Now, if you run __main__.py script following message will show:
    
    Please enter one of the mode:<br>
        train : To train the model <br>
        test: To test the model <br>
        train-test: To train and then test <br> 
        infer: To see result for a single instance <br>
    
    So lets talk how to install DAMCA Recommender for each option.

3. Training:
    1. Download [dataset](https://drive.google.com/open?id=1tRNDwVKYx1cN8R8A0_JXG8bMrz7zGQTn) or create dataset by your own using DAMCA Context Collector. Each sentence need to maintain following structure:<br>
        fileLocation startLOC : endLOC +++$+++ methodCallSequence +++$+++ recieverVariableFQN 
        +++$+++ context
    2. Put the files in /damca_dataset/train_dataset/ folder.
    3. Check and edit(if needed) global variables in config.py and run __main__.py. Write 'train' in the first interface.
    4. The program will train from the input data and will generate a model file, input and output vocabulary.
4. Test:
    1. Download the [Model file](https://drive.google.com/open?id=1sR9tlmABaS36ns16A3T3kk0LSJW5pUFy), [input vocabulary](https://drive.google.com/open?id=1sG98mUD5fFXU3_LNmxCJb8hxvkFUEAti) and [output vocabulary](https://drive.google.com/open?id=1pxeZfOjkWPv30jNhtck1Frz0Nwq6DwOf)
    2. Put input and output vocabulary in /damca_vocabulary folder and model file in the project folder. 
    3. Download [dataset](https://drive.google.com/open?id=1tRNDwVKYx1cN8R8A0_JXG8bMrz7zGQTn) or create dataset by your own using DAMCA Context Collector. 
    4. Put the files in /damca_dataset/test_dataset/ folder.
    5. Check and edit(if needed) global variables in config.py and run __main__.py. Write 'test' in the first interface.
    6. The program will test from the input data and will generate a results in /result folder.
5. Train-test:
    1. Same as Training and after training it will test using the file assigned as a path in config.py as 'test_dataset_file_path'.
6. Infer:
    1. All process is same as Test except you need not download and put dataset in /damca_dataset/test_dataset folder.
    2. After writing 'infer' at the first interface, it will ask to give whole test case in a single line. Paste the following line as a sample input:
    ```/home/local/SAIL/parvezku01/research/parameter_recommendation/repository/eclipse-sourceBuild-srcIncluded-3.7.2/src/plugins/org.eclipse.jdt.ui/ui/org/eclipse/jdt/internal/ui/actions/SelectAllAction.java 14 : 69 +++$+++ getExpanded:org.eclipse.swt.widgets.TreeItem +++$+++ org.eclipse.swt.widgets.TreeItem +++$+++ getData  if if add getData if collectExpandedAndVisible int getData if List for SelectAllAction TreeItem Action```
    3. It will show the top-k suggestion in the console.


## Built With

* [Keras](https://keras.io/) - Python library built on top of tensorflow for deep neural network.
* [TF Estimatot](https://www.tensorflow.org/guide/estimators) - Python library built on top of tensorflow for deep neural network.

## Author

* **C M Khaled Saifullah** - *Initial work* - [khaledkucse](https://github.com/khaledkucse)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details
