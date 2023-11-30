import numpy as np
from keras.preprocessing import image
from keras.models import load_model  # Import load_model directly from keras.models

model_path = 'primarly_test.h5'

# Load the model
loaded_model = load_model(model_path, compile=False)
loaded_model.load_weights('primarly_wights.h5')

def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(180, 300))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    result = model.predict(img)
    return result[0][0]

# Example usage with the loaded model
import os

def predict_images_in_folder(model, folder_path):
    pollen_count = 0
    not_poll = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Assuming all images are JPEG files
            image_path = os.path.join(folder_path, filename)
            prediction = predict_image(model, image_path)
            if prediction >= 0.5:
                pollen_count += 1
            else:
                not_poll += 1
    print(f"Total images with pollen: {pollen_count}")
    print(f"Total images without pollen: {not_poll}")

# Example usage with the loaded model and a folder of images
folder_path = "Data/no_pollen"
predict_images_in_folder(loaded_model, folder_path)
