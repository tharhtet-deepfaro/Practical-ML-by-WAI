import tensorflow as tf
import numpy as np
import json
import os
from dotenv import load_dotenv
load_dotenv()

ML_Summer_School_ID = os.getenv('ML_Summer_School_ID')
print("Your Sudent ID is: " + ML_Summer_School_ID)



test_dir = 'test_dir'
model_path = 'my_model.h5'
class_index_path = 'class_indices.json'


print(os.getcwd())