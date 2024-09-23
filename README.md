**Age Prediction Using CNN**
This repository contains code for predicting age using a Convolutional Neural Network (CNN) model. 

**The project includes two main components:**
A Jupyter notebook (train_model.ipynb) for training the CNN model, saving the trained model as an .h5 file.
A Streamlit app (app.py) that loads the saved model and predicts age from input images.

**Project Overview**
Age prediction from images is a challenging problem in the field of computer vision. This project utilizes a CNN to predict the age of individuals based on facial images. The model is trained on a custom dataset and saved in .h5 format. The trained model is then deployed in a Streamlit app to provide a user-friendly interface for predictions.

**Files:**
**train_model.ipynb:** This Jupyter notebook contains the code for training the CNN model. 
It involves:
Loading the dataset of facial images.
Preprocessing the data (resizing, normalization).
Building and training a CNN model using Keras.
Saving the trained model as age_prediction_model.h5.

**app.py:**
This is a Python file that uses the Streamlit framework to create a web interface for the age prediction model. Users can upload images, and the app will predict the age based on the uploaded image using the trained CNN model.
