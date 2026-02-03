import cv2
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------
# Load trained model
# -----------------------
model = load_model("models/caption_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max_length
with open("max_length.txt", "r") as f:
    max_length = int(f.read().strip())

# -----------------------
# Load CNN for feature extraction
# -----------------------
base_model = InceptionV3(weights="imagenet")
cnn_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("mixed10").output
)

# -----------------------
# Feature extraction from webcam frame
# -----------------------
def extract_feature_from_frame(frame):
    frame = cv2.resize(frame, (299, 299))
    image = img_to_array(frame)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = cnn_model.predict(image, verbose=0)
    feature = feature.reshape((1, 64, 2048))
    return feature.astype("float32")

# -----------------------
# Beam search caption generation
# -----------------------
def generate_caption_beam(feature, beam_width=3):
    start_token = tokenizer.word_index["start"]
    end_token = tokenizer.word_index["end"]

    sequences = [[ [start_token], 0.0 ]]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == end_token:
                all_candidates.append([seq, score])
                continue

            padded_seq = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([feature, padded_seq], verbose=0)[0]

            top_words = np.argsort(preds)[-beam_width:]

            for word_idx in top_words:
                new_seq = seq + [word_idx]
                new_score = score - np.log(preds[word_idx] + 1e-9)
                all_candidates.append([new_seq, new_score])

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    best_sequence = sequences[0][0]

    caption_words = []
    for idx in best_sequence:
        word = tokenizer.index_word.get(idx)
        if word == "end":
            break
        if word != "start":
            caption_words.append(word)

    return " ".join(caption_words)

# -----------------------
# Webcam live captioning
# -----------------------
cap = cv2.VideoCapture(0)

caption = "Starting camera..."
frame_count = 0
CAPTION_INTERVAL = 30  # update caption every 30 frames (~1 sec)

print("🎥 Live captioning started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Update caption periodically (important for CPU)
    if frame_count % CAPTION_INTERVAL == 0:
        feature = extract_feature_from_frame(frame)
        caption = generate_caption_beam(feature, beam_width=3)

    # Display caption
    cv2.putText(
        frame,
        caption,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Image Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
