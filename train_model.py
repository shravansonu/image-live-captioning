import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, Activation, Dot, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


# Load captions
captions = open("dataset/captions.txt").read().strip().split('\n')[1:]

import pickle
features = pickle.load(open("features.pkl", "rb"))

mapping = {}

for line in captions:
    tokens = line.split(',', 1)
    image_id = tokens[0]
    caption = tokens[1].lower()

    if image_id not in features:
        continue   # prevents KeyError

    if image_id not in mapping:
        mapping[image_id] = []

    mapping[image_id].append("start " + caption + " end")

print("Total images used:", len(mapping))

# Load image features
features = pickle.load(open("features.pkl", "rb"))

# Tokenizer
all_captions = []
for caps in mapping.values():
    all_captions.extend(caps)

tokenizer = Tokenizer(oov_token="unk")
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

max_length = max(len(c.split()) for c in all_captions)
with open("max_length.txt", "w") as f:
    f.write(str(max_length))

# DATA GENERATOR (KEY FIX)
def data_generator(mapping, features, tokenizer, max_length, vocab_size, batch_size=32):
    X1, X2, y = [], [], []
    n = 0

    while True:
        for img_id, caps in mapping.items():
            feature = features[img_id].astype("float32")
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical(out_seq, num_classes=vocab_size)

                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1

                    if n == batch_size:
                        yield ([np.array(X1), np.array(X2)], np.array(y))
                        X1, X2, y = [], [], []
                        n = 0

def attention_block(features, hidden):
    """
    features: (batch, 64, 256)
    hidden:   (batch, 256)
    """

    # Expand hidden state to (batch, 1, 256)
    hidden = tf.expand_dims(hidden, axis=1)

    # Dot product attention
    score = Dot(axes=[2, 2])([features, hidden])   # (batch, 64, 1)
    attention_weights = Activation("softmax")(score)

    # Context vector
    context = Dot(axes=[1, 1])([attention_weights, features])  # (batch, 1, 256)
    context = tf.squeeze(context, axis=1)  # (batch, 256)

    return context



# Model
# Image features input (64, 2048)
inputs1 = Input(shape=(64, 2048))
features_dense = Dense(256, activation="relu")(inputs1)

# Caption input
inputs2 = Input(shape=(max_length,))
embedding = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
embedding = Dropout(0.3)(embedding)
lstm_output = LSTM(256)(embedding)

# APPLY ATTENTION HERE
context = attention_block(features_dense, lstm_output)

# Combine attention context and LSTM output
decoder = Add()([context, lstm_output])
decoder = Dense(256, activation="relu")(decoder)
outputs = Dense(vocab_size, activation="softmax")(decoder)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Training
steps = 6000

early_stop = EarlyStopping(
    monitor='loss',
    patience=7,
    verbose=1,
    restore_best_weights=True
)


model.fit(
    data_generator(mapping, features, tokenizer, max_length, vocab_size),
    steps_per_epoch=steps,
    epochs=20,
    callbacks=[early_stop],
    verbose=1
)


model.save("models/caption_model.h5")
print("Model trained successfully using generator")
