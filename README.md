# Vision and Perception Project
Author: Alessio Saladino
## Goal of the project
The goal of the project consist in the classification of actions showed in videos.
In order to accomplish this task the dataset UCF101 have been used.  
## Dataset Generation
Before proceding with the training, it is necessary to generate the samples that will be used.  
In order to do that it is necessary to run the file dataset_handler.py with argument the number of classes desired.  
Eg:  run dataset_handler.py --num_classes 10  
Once the file have been executed, it will take a while to generate a new folder, this folder will contain the tensors of the videos and the train,validation and test splits, that will be used during the training.  
##Train

