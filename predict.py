import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import random

def load_and_process_image_for_prediction(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể tải ảnh từ đường dẫn: {image_path}")
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

def calculate_mean_reconstruction_error(autoencoder, image_paths, sample_size=20):
    # Lấy mẫu ngẫu nhiên từ danh sách các đường dẫn ảnh
    sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))
    
    reconstruction_errors = []
    
    for image_path in sample_paths:
        processed_image = load_and_process_image_for_prediction(image_path)
        reconstructed_image = autoencoder.predict(processed_image)
        error = np.mean(np.abs(processed_image - reconstructed_image))
        reconstruction_errors.append(error)
    
    return np.mean(reconstruction_errors)

def is_water_dispenser(image_path, autoencoder, mean_reconstruction_error, threshold_factor=1.0):
    processed_image = load_and_process_image_for_prediction(image_path)
    reconstructed_image = autoencoder.predict(processed_image)
    
    reconstruction_error = np.mean(np.abs(processed_image - reconstructed_image))
    
    print(f"Sai số tái tạo: {reconstruction_error}")
    
    threshold = mean_reconstruction_error * threshold_factor
    
    if reconstruction_error < threshold:
        return True
    else:
        return False

# Tải mô hình
autoencoder = load_model('autoencoder_model.h5')

def get_image_paths(directory):
    image_paths = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(file_path)

    return image_paths

# Thư mục chứa ảnh
image_directory = "D:/code/BTL_HTTM/data/train"

# Lấy tất cả các đường dẫn ảnh
image_paths_for_mean_error = get_image_paths(image_directory)

# Tính sai số tái tạo trung bình từ tập dữ liệu huấn luyện
mean_reconstruction_error = calculate_mean_reconstruction_error(autoencoder, image_paths_for_mean_error)
print(f'Sai số tái tạo trung bình: {mean_reconstruction_error}')

# Kiểm tra một ảnh cụ thể
test_image_path = "D:/code/BTL_HTTM/data/test/1.jpg"
result = is_water_dispenser(test_image_path, autoencoder, mean_reconstruction_error)

if result:
    print("Ảnh là cây nước!")
else:
    print("Ảnh không phải là cây nước.")