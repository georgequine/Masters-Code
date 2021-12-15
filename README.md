# Masters-Code
An Energy Harvesting Portable and Rollable Large Area Gestural Interface Using Ambient Light


An Energy Harvesting Portable and Rollable Large Area Gestural Interface Using Ambient Light

7 folders
 - Convolutional Neural Network
 - Random Forest
 - Data Pipeline
 - Extra
 - GUI
 - Testing Data_
 - Training Data_


Both Convolutional Neural Network and Random Forest folders contain a model file and test file.
Model file is where the architecture and training of the model in implemented
Test file is where the model is tested with various light itensity datasets 

Data Pipeline contains a file to concentrate all seperate gestures into one large dataframe.
You can upload a folder containing a series of labelled gestures to concetrate. 
The folder to upload must be organised by DATASET NAME -> LABELLED GESTURE FOLDER -> Gesture csv

Extra contains all files needed to run the model and tests. Such as a serialized verson of PCA 

GUI contains a GUI for both models. Run this in command by locating the directory and running python FILE_NAME.py

Testing Data - Contains a 6 different light intensities and one flashing light dataset. All unseen to the model

Training Data - The dataset the model was trained on.



**** ALL CODE WAS IMPLEMENTED ON GOOGLE COLAB (minus Data Pipeline which was implemetned on Jupyter Notebook) *****
To run this code, I highly recommend using Google Colab. It is free
Here is a link to a shared google colab folder with all code and tests available.
Link : https://drive.google.com/drive/folders/1DwjBZtHoHwco2BUn804p5lMKU3-78fuy?usp=sharing
