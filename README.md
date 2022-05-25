# Vision and Perception Project
## Goal of the project
The goal of the project consist in the classification of actions showed in videos.
In order to accomplish this task the dataset UCF101 have been used.  
The classification was made by using Resnet18 as feature extractor for the videos, then those features have been passed as input for a LSTM that have provided the input for the last, linear layers, that provided the final classification.
## Dataset Generation
Before proceding with the training, it is necessary to generate the samples that will be used.  
In order to do that it is necessary to run the file dataset_handler.py with the following arguments:
* data_folder_name: the name of the folder that will be created and that will contain the tensors of the videos;
* n_frames: how many frames will be collected from the videos

Eg: !run dataset_handler.py --data_folder_name FOLDER_NAME --classes Biking-Surfing --n_frames 5

Once the file have been executed, it will take a while to generate all the tensor.
## Train
In order to proceed with the train, it is necessary to run the main.py file by specifing the following training parameters:  
* data_folder_name: the program will automatically load the tensors in the specified folder and it will use them for the training;
* epochs;
* batch_size;
* optimizer;
* learning_rate;
* validation_size.  
 
Eg: !run main.py --data_folder_name FOLDER_NAME --epochs 30 --batch_size 4 --noise 0 --optimizer 'SGD' --learning_rate 0.001 --validation_size 0.2

## Train results
After the training, a new folder in Saves will be created and it will contain the following files:
* Training_parameters.txt: a text file that will recap the parameters used for the training session and the structure of the model
* Model.pth: the trainel model
* Train_history.png, Val_history.png: two graphs that will show the training and loss curves during the training


