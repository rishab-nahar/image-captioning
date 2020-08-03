from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_features(fname, model):
    try:
        img = Image.open(fname)

    except:
        print("ERROR: Couldn't open image")
    img = img.resize((299, 299))
    img = np.array(img)
    if img.shape[2] == 4:
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)
    img = img / 127.5
    img = img - 1.0
    feature = model.predict(img)
    return feature


def word_for_id(n, tokenizer):
    for word, ind in tokenizer.word_index.items():
        if ind == n:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


max_length = 32
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('models/model_24.h5')
xception_model = Xception(include_top=False, pooling="avg")


import re


#large dataset/14919212.jpg
#large dataset/1440465.jpg
img_path = "large dataset/256063.jpg"
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
description = re.sub(r"(red|pink|two|together)", "", description)
description = re.sub(r"ocean", "water", description)
print(description)

plt.imshow(img)
