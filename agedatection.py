import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from tqdm import tqdm
from PIL import Image
import streamlit as st

# Set the base directory to the location of the dataset
BASE_DIR = 'UTKFace2'

# Initialize lists to hold age labels and image paths
age_labels = []
image_paths = []

# Streamlit app title
st.title('Age Prediction from UTKFace Dataset')

# Function to extract image features
def extract_image_features(images):
    features = list()
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

# Function to get image features for prediction
def get_image_features(image):
    img = load_img(image, color_mode='grayscale')
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img / 255.0
    return img

# Load dataset
if st.button('Load Dataset'):
    image_filenames = os.listdir(BASE_DIR)
    random.shuffle(image_filenames)
    for image in tqdm(image_filenames):
        image_path = os.path.join(BASE_DIR, image)
        img_components = image.split('_')
        age_label = int(img_components[0])
        age_labels.append(age_label)
        image_paths.append(image_path)
    
    st.write(f'Number of age_labels: {len(age_labels)}, Number of image_paths: {len(image_paths)}')
    df = pd.DataFrame()
    df['image_path'], df['age'] = image_paths, age_labels
    st.dataframe(df.head(5))

    # Display a random image with its age label
    rand_index = random.randint(0, len(image_paths) - 1)
    age = df['age'][rand_index]
    IMG = Image.open(df['image_path'][rand_index])
    st.image(IMG, caption=f'Age: {age}', use_column_width=True)

    # Plot age distribution
    st.subheader('Age Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['age'], kde=True, ax=ax)
    st.pyplot(fig)

    # Display samples of images and their age labels
    st.subheader('Sample Images')
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    samples = df.iloc[0:16]
    for ax, (index, sample_row) in zip(axes.flatten(), samples.iterrows()):
        img = load_img(sample_row['image_path'])
        img = np.array(img)
        ax.axis('off')
        ax.set_title(f'Age: {sample_row["age"]}')
        ax.imshow(img)
    st.pyplot(fig)

    # Extract features and normalize them
    st.write('Extracting image features...')
    X = extract_image_features(df['image_path'])
    X = X / 255.0
    y_age = np.array(df['age'])

    # Build the model
    input_shape = (128, 128, 1)
    inputs = Input(input_shape)
    conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    max_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(max_1)
    max_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(max_2)
    max_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(max_3)
    max_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    flatten = Flatten()(max_4)
    dense_1 = Dense(256, activation='relu')(flatten)
    dropout_1 = Dropout(0.3)(dense_1)
    output_1 = Dense(1, activation='relu', name='age_out')(dropout_1)

    model = Model(inputs=[inputs], outputs=[output_1])

    # Compile the model using strings for loss and metrics
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    st.write('Model summary:')
    model.summary(print_fn=lambda x: st.text(x))

    # Train the model
    st.write('Training the model...')
    history = model.fit(x=X, y=y_age, batch_size=32, epochs=20, validation_split=0.2)
    
    # Plot training and validation loss
    st.subheader('Training and Validation Loss')
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    fig, ax = plt.subplots()
    ax.plot(epochs, loss, 'b', label='Training Loss')
    ax.plot(epochs, val_loss, 'r', label='Validation Loss')
    ax.set_title('Loss Graph')
    ax.legend()
    st.pyplot(fig)

    # Save the trained model
    model.save('age_prediction_model.h5')
    st.write('Model saved as age_prediction_model.h5')

# Upload and predict age for a new image
uploaded_file = st.file_uploader('Upload an image to predict age', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Check if the model has been saved previously
    if not os.path.exists('age_prediction_model.h5'):
        st.error('Model has not been trained yet. Please load and train the dataset first.')
    else:
        # Load the saved model without specifying custom objects
        model = load_model('age_prediction_model.h5')
        
        img_to_test = uploaded_file
        features = get_image_features(img_to_test)
        pred = model.predict(features)
        age = round(pred[0][0])
        st.image(img_to_test, caption=f'Predicted Age: {age}', use_column_width=True)
        
        # Display the predicted age in a larger font
        st.markdown(f'<h2 style="text-align: center; color: blue;">Predicted Age: {age}</h2>', unsafe_allow_html=True)
