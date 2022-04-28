# Vision and Perception Project
## Goal of the project
The goal of the project consist in the classification of actions showed in videos.
In order to accomplish this task the dataset UCF101 have been used.  
## Dataset Generation
Before proceding with the training, it is necessary to generate the samples that will be used.  
In order to do that it is necessary to run the file dataset_handler.py with argument the number of classes desired.  
Eg: !run dataset_handler.py --num_classes 10  
Once the file have been executed, it will take a while to generate a new folder, this folder will contain the tensors of the videos and the train,validation and test splits, that will be used during the training.  
## Train
In order to proceed with the train, it is necessary to run the main.py file by specifing the following training parameters:  
* num_classes
* epochs
* batch_size
* criterion
* optimizer
* learning_rate
* validation_size  
Eg: !run main.py --num_classes 10 --epochs 10 --batch_size 4 --criterion 'CrossEntropy' --optimizer 'SGD' --learning_rate 0.001 --validation_size 0.2  

After the training, a new folder in Saves will be created and it will contain the following files:
* Training_parameters.txt: a text file that will recap the parameters used for the training session and the structure of the model
* Model.pth: the trainel model
* Train_history.png, Val_history.png: two graphs that will show the training and loss curves during the training
