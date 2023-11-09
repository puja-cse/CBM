### Interactive learning of concept based models 

In this project we have implemented Concept Bottlenect Model (CBM) with Active Learning. The neural network architecure is displayed in the following Figure.

![image](/diagrams/img-model.jpg)  

<br>
There are five folders containing different files: <br>
- dataset: contains the CUB dataset and the pickle file of the train-test split that we have used in our project <br>
-source: contains the methods to define model structure, training ans testing loop <br>
- diagrams: diagrams related to our project
- saved models: we have experimented with different values of Lambda, and this folder contains all the trained models that we generated during the experiment <br>
- output files: this folder contains the output file for each model and each output file containes detailed result of each experiment <br> 

<br>
In the jupyter_notebooks folder,   
Train_CBM_CUB.ipynb contains the required step to train the CBM model and Test_CBM_CUB.ipynb file contains the instructions involved in testing the model with the test dataset.
<br>  
CUB_AL.ipynb file contains the code segment of active learning framework and the output for active learning.

