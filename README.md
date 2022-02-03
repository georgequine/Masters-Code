# An Energy Harvesting Portable and Rollable Large Area Gestural Interface Using Ambient Light

In this work, we explored the viability of using ambient light and a large photovoltaic sheet for energy harvesting
and gesture recognition. Our prototype consists of a large, portable, and rollable gestural interface which can
uniquely distinguish and classify distinct hand gestures performed by a user. The system works under the principle
that the amount of power harvested by the photodiodes decreases when a near-field object blocks the surrounding
light. By monitoring these fluctuations, we recognised that different shadow patterns produce a distinct signature in
the amplitude of the harvested voltage. Delegating the detection responsibilities to machine learning, it was possible
to capture the hidden meaning within the hand gestures to perform an action in real time. We focused on two
classifiers, one utilising a machine learning technique, Random Forest (RF), and the other a deep learning classifier,
a Convolutional Neural Network (CNN). To further improve the robustness of the system, we applied two preprocessing 
techniques known as Normalisation and Principal Component Analysis to reduce inherent noise caused
by inevitable environmental and human factors. We evaluated the proposed system under a variety of lighting
conditions, as well as assessing the significance of the two pre-processing techniques. We trained our models with
1,050 incidents of 5 unique gestures. The CNN demonstrated the highest overall accuracy in all lighting conditions,
with 95% accuracy in 1K lux. The RF performed similarly well, obtaining 93% accuracy in 1K lux. Using a
designed Graphical User Interface (GUI), both models are capable of recognising an unseen gesture in 0.05 seconds

7 folders
 - Convolutional Neural Network
 - Random Forest
 - Data Pipeline
 - Extra
 - GUI
 - Testing Data_
 - Training Data_


Both Convolutional Neural Network and Random Forest folders contain a model file and test file.

- 'Model' is where the architecture and training of the model in implemented.

- 'Test' is where the model is tested with various light itensity datasets. 

'Data Pipeline' contains a file to concentrate all seperate gestures into one large dataframe.
You can upload a folder containing a series of labelled gestures to concetrate into one dataframe. 
The folder to upload must be organised by SESSION NAME -> THE GESTURE LABEL -> GESTURE.csv

'Extra' contains all files needed to run the model and tests. Such as a serialized verson of PCA 

'GUI' contains a GUI for both models. Run this in command by locating the directory and running python FILE_NAME.py

'Testing Data' contains a dataset of 6 different light intensities and one flashing light. All unseen to the trained model

Training Data - The dataset the model was trained with.



**** ALL CODE WAS IMPLEMENTED ON GOOGLE COLAB (minus Data Pipeline which was implemetned on Jupyter Notebook) *****

To run this code, I highly recommend using Google Colab. It is free

All models and tests contain a link to Google Colab

Here is a link to a shared Google Colab folder with all code and tests available.

Link : https://drive.google.com/drive/folders/1DwjBZtHoHwco2BUn804p5lMKU3-78fuy?usp=sharing
