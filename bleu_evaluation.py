import pickle
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# Load tokenizer, model, features
# -----------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

from tensorflow.keras.models import load_model
model = load_model("models/caption_model.h5")

with open("max_length.txt", "r") as f:
    max_length = int(f.read().strip())

# -----------------------
# Rebuild mapping (GROUND TRUTH)
# -----------------------
captions = open("dataset/captions.txt", encoding="utf-8").read().strip().split("\n")[1:]

mapping = {}

for line in captions:
    tokens = line.split(",", 1)
    image_id = tokens[0]
    caption = tokens[1].lower()

    if image_id not in features:
        continue

    if image_id not in mapping:
        mapping[image_id] = []

    mapping[image_id].append(caption)

print("Total images with captions:", len(mapping))

# -----------------------
# Beam search caption generator
# -----------------------
def generate_caption_beam(feature, beam_width=3):
    start_token = tokenizer.word_index["start"]
    end_token = tokenizer.word_index["end"]

    sequences = [[[start_token], 0.0]]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == end_token and len(seq) > 5:
                all_candidates.append([seq, score])
                continue

            padded = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([feature.reshape(1,64,2048), padded], verbose=0)[0]

            top_words = np.argsort(preds)[-10:]

            for w in top_words:
                new_seq = seq + [w]
                new_score = (score - np.log(preds[w] + 1e-9)) / len(new_seq)
                all_candidates.append([new_seq, new_score])

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    best = sequences[0][0]

    caption = []
    for idx in best:
        word = tokenizer.index_word.get(idx)
        if word == "end":
            break
        if word != "start":
            caption.append(word)

    return " ".join(caption)

# -----------------------
# Train-test split
# -----------------------
image_ids = list(mapping.keys())
split = int(0.8 * len(image_ids))
test_images = image_ids[split:]

# -----------------------
# BLEU Evaluation
# -----------------------
actual, predicted = [], []
smoothie = SmoothingFunction().method4

for img_id in test_images:
    refs = [cap.split() for cap in mapping[img_id]]

    y_pred = generate_caption_beam(features[img_id])

    if y_pred is None or len(y_pred.split()) < 2:
        continue

    actual.append(refs)
    predicted.append(y_pred.split())

print("Images evaluated:", len(predicted))

print("BLEU-1:", corpus_bleu(actual, predicted, weights=(1,0,0,0), smoothing_function=smoothie))
print("BLEU-2:", corpus_bleu(actual, predicted, weights=(0.5,0.5,0,0), smoothing_function=smoothie))
print("BLEU-3:", corpus_bleu(actual, predicted, weights=(0.33,0.33,0.33,0), smoothing_function=smoothie))
print("BLEU-4:", corpus_bleu(actual, predicted, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie))