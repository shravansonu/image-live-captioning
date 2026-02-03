import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Pretrained model (BLIP)
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Text to Speech
import pyttsx3

# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------
# Load trained caption model
# -----------------------
model = load_model("models/caption_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# max_length (safe fallback)
try:
    with open("max_length.txt") as f:
        max_length = int(f.read().strip())
except:
    max_length = 34

# -----------------------
# CNN feature extractor
# -----------------------
cnn_model = InceptionV3(weights="imagenet")
cnn_model = torch.nn.Sequential() if False else cnn_model
cnn_model = load_model("inception_feature_extractor.h5") if os.path.exists(
    "inception_feature_extractor.h5"
) else cnn_model

feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling=None)

# -----------------------
# Pretrained BLIP model
# -----------------------
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model.eval()

# -----------------------
# Text-to-Speech
# -----------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)
tts_engine.setProperty("volume", 1.0)

def speak_text(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except:
        pass

# -----------------------
# Feature extraction
# -----------------------
def extract_feature(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = feature_extractor.predict(image, verbose=0)
    features = features.reshape((1, 64, 2048))
    return features

# -----------------------
# Beam Search Caption (OUR MODEL)
# -----------------------
def generate_caption_beam(feature, beam_width=3):
    start = tokenizer.word_index["start"]
    end = tokenizer.word_index["end"]

    sequences = [[[start], 0.0]]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == end:
                all_candidates.append((seq, score))
                continue

            padded = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([feature, padded], verbose=0)[0]

            top = np.argsort(preds)[-beam_width:]

            for word in top:
                candidate = seq + [word]
                candidate_score = score - np.log(preds[word] + 1e-10)
                all_candidates.append((candidate, candidate_score))

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    final_seq = sequences[0][0]
    caption = []

    for i in final_seq:
        word = tokenizer.index_word.get(i)
        if word in ["start", "end"]:
            continue
        caption.append(word)

    return " ".join(caption)

# -----------------------
# Greedy Caption (LESS REPETITIVE)
# -----------------------
def generate_caption_greedy(feature):
    start = tokenizer.word_index["start"]
    end = tokenizer.word_index["end"]

    seq = [start]

    for _ in range(max_length):
        padded = pad_sequences([seq], maxlen=max_length)
        preds = model.predict([feature, padded], verbose=0)[0]

        next_word = np.argmax(preds)
        seq.append(next_word)

        if next_word == end:
            break

    caption = []
    for i in seq:
        word = tokenizer.index_word.get(i)
        if word not in ["start", "end"]:
            caption.append(word)

    return " ".join(caption)


# -----------------------
# Pretrained Caption (BLIP)
# -----------------------
def generate_caption_pretrained(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")

    with torch.no_grad():
        output = blip_model.generate(**inputs, max_length=30, num_beams=5)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

# -----------------------
# Flask Route
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    caption_ours = None
    caption_pretrained = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            feature = extract_feature(image_path)
            caption_ours = generate_caption_greedy(feature)
            caption_pretrained = generate_caption_pretrained(image_path)
            image_url = image_path

            # 🔊 Speak caption (our model)
            speak_text("caption generated is: " + caption_pretrained) #"Our model says: " + caption_ours +

    return render_template(
        "index.html",
        #caption_ours=caption_ours,
        caption_pretrained=caption_pretrained,
        image_url=image_url
    )

# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
