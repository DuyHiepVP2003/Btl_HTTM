import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

# Hàm load và xử lý ảnh
def load_and_process_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể tải ảnh từ đường dẫn: {image_path}")
    img = cv2.resize(img, target_size)
    return img / 255.0  # Chuẩn hóa giá trị pixel

# Hàm xử lý ảnh trong thư mục
def process_images_in_directory(directory, target_size=(64, 64)):
    processed_images = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                processed_image = load_and_process_image(file_path, target_size)
                processed_images.append(processed_image)
            except ValueError as e:
                print(e)
    return processed_images

# Load và xử lý dữ liệu ảnh
image_directory = "D:/code/BTL_HTTM/data/train"
processed_images = process_images_in_directory(image_directory, target_size=(64, 64))
processed_images = np.array(processed_images)

# Định nghĩa Autoencoder
input_img = Input(shape=(64, 64, 3))  
x = Conv2D(48, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(32, (1, 1), activation='relu', padding='same')(x)

latentSize = (8, 8, 32)

# Decoder
direct_input = Input(shape=latentSize)
x = Conv2D(192, (1, 1), activation='relu', padding='same')(direct_input)
x = UpSampling2D((2, 2))(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Compile Autoencoder
encoder = Model(input_img, encoded)
decoder = Model(direct_input, decoded)
autoencoder = Model(input_img, decoder(encoded))
autoencoder.summary()

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

# Huấn luyện Autoencoder và lưu kết quả huấn luyện
history = autoencoder.fit(processed_images, processed_images, epochs=50, batch_size=32, validation_split=0.1)

# Lưu mô hình đã huấn luyện
autoencoder.save('autoencoder_model.h5')

# Lưu kết quả huấn luyện (loss, accuracy, v.v.) vào file CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

print("Đã lưu mô hình vào 'autoencoder_model.h5' và kết quả huấn luyện vào 'training_history.csv'.")

# Vẽ biểu đồ loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()