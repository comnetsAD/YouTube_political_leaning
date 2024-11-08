# Import necessary libraries
from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import tensorflow_text as text
import numpy as np
from nltk.corpus import stopwords
import nltk
import re

# Load datasets
tr = pd.read_csv('train.csv')  # Training dataset
val = pd.read_csv('valid.csv')  # Validation dataset
te = pd.read_csv('test.csv')  # Test dataset

# Separate features (titles) and labels for training, validation, and test sets
X_train = tr['title']
y_train = tr['label']
X_val = val['title']
y_val = val['label']
X_test = te['title']
y_test = te['label']

# Convert data to lists for easier manipulation
X_train2 = list(np.asarray(X_train.values))
y_train2 = list(np.asarray(y_train.values))
X_val2 = list(np.asarray(X_val.values))
y_val2 = list(np.asarray(y_val.values))
X_test2 = list(np.asarray(X_test.values))
y_test2 = list(np.asarray(y_test.values))

# Ensure labels are integers for compatibility with the model
y_train2 = [int(x) for x in y_train2]
y_val2 = [int(x) for x in y_val2]
y_test2 = [int(x) for x in y_test2]

# Define a checkpoint to save the best model based on validation loss
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5', monitor='val_loss', 
    save_best_only=True, save_weights_only=False, mode='min'
)

# Calculate class weights to handle class imbalance
total = len(y_train2)
tot_0 = y_train2.count(0)
tot_1 = y_train2.count(1)
tot_2 = y_train2.count(2)
tot_3 = y_train2.count(3)
tot_4 = y_train2.count(4)
tot_5 = y_train2.count(5)

# Define weights for each class based on class frequencies
w0 = total / (6 * tot_0)
w1 = total / (6 * tot_1)
w2 = total / (6 * tot_2)
w3 = total / (6 * tot_3)
w4 = total / (6 * tot_4)
w5 = total / (6 * tot_5)

class_weight = {0: w0, 1: w1, 2: w2, 3: w3, 4: w4, 5: w5}

# Use MirroredStrategy for distributed training across multiple GPUs
strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])

with strategy.scope():
    # Load pre-trained BERT model and preprocess layer from TensorFlow Hub
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=True)
    
    # Define model architecture
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)  # Preprocess text input
    outputs = bert_encoder(preprocessed_text)  # Encode text using BERT
    
    # Apply dropout and dense layers
    l = tf.keras.layers.Dropout(0.3)(outputs['pooled_output'])
    l = tf.keras.layers.Dense(512, activation='relu')(l) 
    l = tf.keras.layers.Dropout(0.3)(l) 
    l = tf.keras.layers.Dense(6, activation='softmax')(l)  # Output layer for 6 classes
    
    # Build the model
    model = tf.keras.Model(inputs=[text_input], outputs=[l])
    model.summary()  # Print model summary
    
    # Compile model with Adam optimizer and sparse categorical crossentropy loss
    opt = tf.keras.optimizers.Adam(learning_rate=1e-04)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train model with class weights and checkpointing
    model.fit(
        X_train2, y_train2, epochs=10, batch_size=128,
        validation_data=(X_val2, y_val2),
        callbacks=[checkpoint],
        class_weight=class_weight
    )

    # Load the best saved model
    model = load_model("best_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})

    # Evaluate model on test data
    print(model.evaluate(X_test2, y_test2, batch_size=1024))

    # Generate predictions on test data
    y_predicted = model.predict(X_test2)

# Convert predictions to label indices
y_predicted = np.argmax(y_predicted, axis=1)
