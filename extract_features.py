import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# 🔧 CPU tuning (important for 8GB RAM)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

IMAGE_DIR = "dataset/Images"
FEATURE_FILE = "features.pkl"
SAVE_EVERY = 300
MAX_IMAGES = 24000   #  your chosen limit

features = {}
print(f"Starting feature extraction (max {MAX_IMAGES} images)")

# Load CNN
base_model = InceptionV3(weights="imagenet")
cnn = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("mixed10").output
)

image_files = os.listdir(IMAGE_DIR)
processed_since_save = 0

for img_name in image_files:
    if len(features) >= MAX_IMAGES:
        print(f"Reached {MAX_IMAGES} images. Stopping extraction.")
        break

    try:
        path = os.path.join(IMAGE_DIR, img_name)
        image = load_img(path, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        feature = cnn.predict(image, verbose=0)
        feature = feature.reshape((64, 2048)).astype("float16")  # memory-safe

        features[img_name] = feature
        processed_since_save += 1

        if processed_since_save >= SAVE_EVERY:
            with open(FEATURE_FILE, "wb") as f:
                pickle.dump(features, f)
            print(f"Saved {len(features)} features")
            processed_since_save = 0

    except Exception as e:
        print(f"Skipping {img_name}")

# Final save
with open(FEATURE_FILE, "wb") as f:
    pickle.dump(features, f)

print("Feature extraction finished")
print(f"Total features saved: {len(features)}")