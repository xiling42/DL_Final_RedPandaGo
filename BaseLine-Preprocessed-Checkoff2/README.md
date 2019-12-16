# DL_Final_RedPandaGo Check-off-2

This is our baseline implementation for baseline.
## Folder
### processed_data: 
folder that contains all our data saved in .npy format after transpose. 
There are 440 labels and 440 data files. Every label and data file should contain 10000 rows. 

## Files
### train.mat
Train file that we used for this project. 
Contains 'traindata' for data and 'trainxdata' for labels.
Data has size: 1000, 4, 4400000
Labels has size: 919, 4400000

### test.mat
Test file.

### data_process.ipynb: 
To process train.met data.

### baseline_model.ipynb:
CNN model we implemented for baseline purpose. It should contain the model class, training, testing and main. We used Pytorch.
	
	
