import tensorflow as tf
import numpy as np
import json
import os
from dotenv import load_dotenv
load_dotenv()

ML_Summer_School_ID = os.getenv('ML_Summer_School_ID')
print("Your Sudent ID is: " + ML_Summer_School_ID)

BATCH_SIZE  = 32
input_img_size = (128,128)
input_shape = (128,128,3)

test_dir = r'/Users/tharhtet/Documents/github/Practical-ML-by-WAI/6_deep_learning/CNN/cats_and_dogs_filtered/test'
model_path = f"./binary_classification/{ML_Summer_School_ID}_model.h5"
class_index_path = f'./binary_classification/{ML_Summer_School_ID}_class_indices.json'


model = tf.keras.models.load_model(model_path)
print(model.summary())

# Load class indices from the JSON file
with open(class_index_path, 'r') as f:
    class_indices = json.load(f)
print("Class indices loaded:", class_indices)




test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=input_img_size,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Important: preserve label order for evaluation
    label_mode='int'  # returns integer labels
)


# === Predict ===
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)

    # If binary classification
    if predictions.shape[1] == 1:
        preds = (predictions > 0.5).astype(int).flatten()
    else:  # multi-class
        preds = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# === Evaluate Accuracy ===
y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = np.mean(y_true == y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")