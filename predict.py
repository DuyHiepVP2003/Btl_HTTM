import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
def load_and_process_image_for_prediction(image_path, target_size=(28, 28)):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path)
    
    # Kiểm tra xem ảnh có được tải không
    if img is None:
        raise ValueError(f"Không thể tải ảnh từ đường dẫn: {image_path}")

    # Chuyển đổi kích thước ảnh
    img = cv2.resize(img, target_size)

    # Chuyển đổi ảnh sang màu xám
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chuyển đổi thành ma trận điểm ảnh và thêm một chiều cho kênh
    img_array = np.array(img).reshape((target_size[0], target_size[1], 1))  # Thêm chiều kênh

    # Chuẩn hóa giá trị pixel
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)  # Thêm chiều batch

def is_water_dispenser(image_path, autoencoder, threshold=0.01):
    # Tiền xử lý ảnh để phù hợp với đầu vào của mô hình
    processed_image = load_and_process_image_for_prediction(image_path)
    
    # Dự đoán ảnh tái tạo từ autoencoder
    reconstructed_image = autoencoder.predict(processed_image)
    
    # Tính sai số tái tạo giữa ảnh gốc và ảnh tái tạo
    reconstruction_error = np.mean(np.abs(processed_image - reconstructed_image))
    
    print(f"Sai số tái tạo: {reconstruction_error}")
    
    # So sánh sai số tái tạo với ngưỡng đã xác định để quyết định
    if reconstruction_error < threshold:
        return True  # Ảnh có thể là cây nước
    else:
        return False  # Ảnh không phải là cây nước

autoencoder = load_model('autoencoder_model.h5')
test_image_path = "D:/code/BTL_HTTM/data/14.jpg"
# Kiểm tra xem ảnh có phải là cây nước hay không
result = is_water_dispenser(test_image_path, autoencoder)

if result:
    print("Ảnh là cây nước!")
else:
    print("Ảnh không phải là cây nước.")