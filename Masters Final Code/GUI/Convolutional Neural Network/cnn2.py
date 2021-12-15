print("Booting System")

# import necessary modules
from tkinter import * 
from tkinter import ttk
from PIL import Image, ImageTk
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle

#Window Customisation
root = Tk()
root.title("Convolutional Neural Network Application")
root.geometry("400x400") # set starting size of window
root.maxsize(400, 400) # width x height
root.config(bg="#6FAFE7") # set background color of root window
 
#Top Header
Heading = Label(root, text="Demonstrator Application", bg="#2176C1", fg='white', relief=RAISED)
Heading.pack(ipady=5, fill='x')
Heading.config(font=("Font", 20)) # change font and size of label

#CNN Model
class Simple1DCNN(torch.nn.Module):
    def __init__(self,cnn_1):
        super(Simple1DCNN, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=14, out_channels=cnn_1, kernel_size=1, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=cnn_1, out_channels=5, kernel_size=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)

        log_probs = torch.nn.functional.log_softmax(x, dim=1)

        return log_probs


#Initialise variables 
list = ["Anti-Clockwise", "Clockwise","Double Hand", "Left Swipe", "Right Swipe"]
fulldata = pd.read_csv('/Users/Owner/Documents/myModel/fulldata_norm.csv')
fulldata = fulldata.iloc[: , 1:]
fulldata.columns = fulldata.columns.map(str)
light1status = 0
global model
model = torch.load('/Users/Owner/Documents/myModel/cnn_norm_FINAL.pt',map_location=device) #Loading in Model via torch
global features
global x
global y 
global principalComponents
global pca
pca = PCA(n_components=0.99)
features = [str(i) for i in range(0,12000)]
x = fulldata.loc[:, features].values
y = fulldata.loc[:,['label']].values
principalComponents = pca.fit_transform(x)
degrees = 0
slider_pos = 0
gesture_number = 0

print("System Ready")


#Switch to control functions depending on detection (Not in use yet)
def switch(detection):

	global gesture	
	global degrees
	global slider_pos

	if detection == "Clockwise":
		gesture = "Clockwise"
		degrees += 25
		rotate_image(degrees)
		print(degrees)



	elif detection == "Anti-Clockwise":
		gesture = "Anti-Clockwise"
		degrees -= 25
		rotate_image(degrees)


	elif detection == "Left Swipe":
		gesture = "Left Swipe"
		slider_pos = -75
		slider(slider_pos)

	elif detection == "Right Swipe":
		gesture = "Right Swipe"
		slider_pos = 75
		slider(slider_pos)

	elif detection == "Double Hand":
		gesture = "Select"
		light1()
	

#Timer at top of window. Once timer reaches 0, user can perform gesture (Not in use yet)
def go():
	# call countdown first time    
	countdown(5)
	# root.after(0, countdown, 5)

def countdown(count):
	# change text in label        
	clock['text'] = count

	if count > 0:
		# call countdown again after 1000ms (1s)
		root.after(1000, countdown, count-1)
	else:
		print("Timer Finished")

#Clock Label
clock = Label(root, text = 'Click begin to start timer')
clock.pack(ipady=5, fill='x', anchor = 'e')

#Button to begin timer
begin = Button(root, text='Begin', width=25, bg = "white", command=go)
begin.pack()

#MyDatset1 class 
class MyDataset1(torch.utils.data.Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.X = torch.unsqueeze(torch.tensor(df.iloc[:,:].values, dtype=torch.float32), dim=-1)
    
    def __getitem__(self, idx):
        return self.X[idx]


#Here we open file and run machine learning model
def open_file():
	global fulldata
	import re

	print("1. Select File")
	rep = filedialog.askopenfilenames()
	tic = time.perf_counter() #Starting timer for 1
	rep = re.sub("[()]","", str(rep))
	#print("Filepath:",rep)
	#rep = rep.replace("/", "\\")
	rep = rep.replace("'", "")
	rep = rep.replace(",", "")
	path=os.path.dirname(rep)
	foldername = os.path.basename(path)
	test_csv = pd.read_csv(str(rep), header=0, skip_blank_lines=True)
	toc = time.perf_counter() #Finishing timer for 1
	print(f"Reading file took {toc - tic:0.4f} seconds")

	print("2. Tranforming the csv")
	tic2 = time.perf_counter() #Starting timer for 2
	Df = test_csv
	# dropping first row as it shows unit
	Df.drop([Df.index[0]], inplace=True)
	Df = Df[1000:13000]
	# converting everything from string to float
	Df = Df.applymap(lambda x : float(x))
	Df['EWM_mean'] = Df.iloc[:,1].ewm(span=200,adjust=True).mean()
	Df.drop(['Time', 'Channel A'], inplace=True, axis=1)
	nw_df = Df.T
	toc2 = time.perf_counter() #Finishing timer for 2
	print(f"Transforming the csv took {toc2 - tic2:0.4f} seconds")

	print("3. Applying MinMax Norm")
	tic3 = time.perf_counter() #Starting timer for 3
	Complete_x = nw_df.iloc[0].to_numpy() 
	Complete_x = Complete_x.reshape(-1,1)
	scaler2 = MinMaxScaler()
	norm = scaler2.fit_transform(Complete_x)
	data_array = norm
	data_array = data_array.flatten()
	in_data = pd.DataFrame(data_array.reshape(-1, len(data_array)))
	in_data.columns = in_data.columns.map(str)
	in_data =  in_data.iloc[0].to_numpy()
	in_data = in_data.reshape(1,-1)
	toc3 = time.perf_counter() #Finishing timer for 4
	print(f" Applying MinMac Norm took {toc3 - tic3:0.4f} seconds")

	print("4. Applying PCA")
	tic4 = time.perf_counter() #Starting timer for 4
	newdata_transformed = pca.transform(in_data)
	pc = pca.explained_variance_ratio_.cumsum()
	x1 = StandardScaler().fit(principalComponents)
	X1 = x1.transform(newdata_transformed)
	newdf = pd.DataFrame(data = X1, columns = [f'pc_stdscaled_{i}' for i in range(len(pc))])
	toc4 = time.perf_counter() #Finishing timer for 4
	print(f"Applying PCA took {toc4 - tic4:0.4f} seconds")


	print("5. MyDatset1 class")
	tic5 = time.perf_counter() #Starting timer for 5
	data = MyDataset1(newdf)
	toc5 = time.perf_counter() #Finishing timer for 5
	print(f"MyDataset1 took {toc5 - tic5:0.4f} seconds")

	print("6. Model evaluate")
	tic6 = time.perf_counter() #Starting timer for 6
	test_loader = torch.tensor(np.expand_dims(data[0],axis=0))
	model.eval()
	toc6 = time.perf_counter() #Finishing timer for 6
	print(f"Model eval took {toc6 - tic6:0.4f} seconds")

	print("7. Predict")
	tic7 = time.perf_counter() #Starting timer for 7
	op = model(test_loader)
	predicted_classes = torch.max(op, 1)[1]
	#print(predicted_classes.item())
	gesture_list = ["Anti-Clockwise", "Clockwise","Double Hand", "Left Swipe", "Right Swipe"]
	prediction = gesture_list[predicted_classes.item()]
	#print("Prediction: ",prediction)
	update(prediction)
	detection = prediction
	switch(detection)
	toc7 = time.perf_counter() #Finishing timer for 7
	print(f"Model Predict took {toc7 - tic7:0.4f} seconds")

	print("")
	print(f"Total time {toc7 - tic:0.4f} seconds")
	print()
	print("Folder Name: ",foldername)
	print("Prediction: ",prediction)
	print("Detection: ",detection)

	if (prediction[0].lower() == foldername[0]):
		print("PASSED: Correct Detection")
	else:
		print("FAILED: Incorrect Detection")


#Button to open file
file = Button(root, text='Open File', width=25, bg = "white", command=open_file)
file.pack()

#Dial knob for rotating once clockwise/anticlockwise has been detected
image = Image.open("dial_knob2.png")
width, height = image.size
image.thumbnail((width/5, height/5))
photoimage = ImageTk.PhotoImage(image)
image_label = Label(root, image=photoimage, bg="#6FAFE7", relief=FLAT)
image_label.image = photoimage
image_label.pack(padx=10, pady=10)

#To rotate file
def rotate_image(degrees):
	new_image = image.rotate(-int(float(degrees)))
	photoimage = ImageTk.PhotoImage(new_image)
	image_label.image = photoimage #Prevent garbage collection
	image_label.config(image = photoimage)
	print("rotate complete")
	
def slider(slider_pos):
	current = w2.get()
	print(current)
	move = current + slider_pos
	w2.set(move)


#Slider once swipe left/right has been detected
w2 = ttk.Scale(root, from_=0, to=360, orient=HORIZONTAL, length=300)
w2.pack(padx=10, pady=10)
w2.set(180)

# #Currently creating 
# def detectionFunc():
# 	global detection
# 	detection = random.choice(list)


#Change colour of select button
def light1():
	global light1status, strongboxit
	if light1status == 0:
		button.config(bg = "red")
		light1status = 1
		#detectionFunc()
		# print(detection)
		# switch()
		# print(gesture)
		# update(detection)


		# update()
		# print(detection)

	else:
		light1status = 0
		button.config(bg = "green")
		# detectionFunc()
		# print(detection)
		# switch()
		# print(gesture)
		# update(detection)
		# update(detection)
		# print(detection)


#Select button for double hand detection 
button = Button(root, text='Select', width=25, bg = "green", command=light1)
button.pack(padx=10, pady=10)

#Update text depending on detetion
def update(detection):
	global gesture_number
	gesture_number += 1
	Heading.config(text = str(gesture_number) + '.' + detection)

#Display text of gesture
Heading = Label(root, text="hello", bg="#2176C1", fg='white', relief=RAISED)
Heading.pack(side="bottom",ipady=5, fill='x')
Heading.config(font=("Font", 20)) # change font and size of label


root.mainloop()