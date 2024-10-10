import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np


def load_and_process_image(image_path, target_size=(28, 28)):
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

    return img_array / 255.0  # Chuẩn hóa giá trị pixel

def process_images_in_directory(directory, target_size=(28, 28)):
    # Danh sách để lưu trữ các ảnh đã được xử lý
    processed_images = []
    
    # Lặp qua tất cả các tệp trong thư mục
    for filename in os.listdir(directory):
        # Xây dựng đường dẫn đầy đủ đến tệp
        file_path = os.path.join(directory, filename)
        
        # Kiểm tra xem tệp có phải là ảnh không (có thể thêm nhiều đuôi tệp khác nếu cần)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Tiến hành tiền xử lý ảnh
                processed_image = load_and_process_image(file_path, target_size)
                processed_images.append(processed_image)

                # Hiển thị ảnh (nếu cần)
                # plt.imshow(processed_image, cmap='gray')
                # plt.axis('off')
                # plt.title(filename)  # Tên tệp ảnh
                # plt.show()
            except ValueError as e:
                print(e)

    return processed_images

# Đường dẫn tới thư mục chứa ảnh
image_directory = "D:/code/BTL_HTTM/data"
processed_images = process_images_in_directory(image_directory, target_size=(28, 28))
# Chuyển đổi danh sách ảnh thành mảng NumPy để đưa vào mô hình
processed_images = np.array(processed_images)
# Kiểm tra kích thước của các ảnh đã xử lý
print(f"Số lượng ảnh đã xử lý: {len(processed_images)}")

encoder_input = keras.Input(shape=(28,28,1), name="encoder_img")
x = keras.layers.Flatten()(encoder_input)
encoder_output = keras.layers.Dense(64, activation="relu")(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")

decoder_input = keras.layers.Dense(784, activation="relu")(encoder_output)
decoder_output = keras.layers.Reshape((28,28,1))(decoder_input)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(processed_images, processed_images, epochs=3, batch_size=32, validation_split=0.1)

autoencoder.save('autoencoder_model.h5')
