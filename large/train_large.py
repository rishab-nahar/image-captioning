
##the dataset can be found at  https://www.kaggle.com/hsankesara/flickr-image-dataset
##importing the keras module
from keras.applications.xception import preprocess_input
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.applications.xception import Xception
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

##other imports
import string
from PIL import Image
import os
from pickle import dump, load
import numpy as np

# Loading a text file into memory
def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def cleaning_text(descriptions):
    table = str.maketrans('', '', string.punctuation)

    for imgs in descriptions.keys():
        for i in range(len(descriptions[imgs])):
            comments = descriptions[imgs][i]
            desc = comments.strip().split(" ")
            desc = [w.lower() for w in desc]
            desc = [w.translate(table) for w in desc]
            desc = [w for w in desc if (len(w) > 1)]
            desc = [w for w in desc if (w.isalpha())]
            desc = [w for w in desc if w != "..."]
            comments = ' '.join(desc)
            descriptions[imgs][i] = comments

    return descriptions


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


def text_vocabulary(descriptions):
    # build vocabulary of all unique words
    vocab = set()

    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]

    return vocab


dataset_images = "large dataset"
import pandas as pd

captions = pd.read_csv("results.csv", delimiter="|")

from collections import defaultdict

descriptions = defaultdict(list)
for i in range(158915):
    descriptions[captions["image_name"][i]].append(str(captions[" comment"][i]))

descriptions = dict(descriptions)
clean_descriptions = cleaning_text(descriptions)

# building vocabulary
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))

# saving each description to file
save_descriptions(clean_descriptions, "descriptions.txt")


def extract_features(directory):
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        # image = preprocess_input(image)
        image = image / 127.5
        image = image - 1.0

        feature = model.predict(image)
        features[img] = feature
    return features


# 2048 feature vector
features = extract_features(dataset_images)
dump(features, open("features.p", "wb"))

features = load(open("features.p", "rb"))

import random

train_imgs = []
for i in clean_descriptions:
    k = random.randint(0, 10)
    if k < 8:
        train_imgs.append(i)

def load_clean_descriptions(filename, photos):
    # loading clean_descriptions
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):

        words = line.split()
        if len(words) < 1:
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)

    return descriptions

def load_features(photos):
    # loading all features
    all_f = load(open("features.p", "rb"))
    # selecting only teh required features
    f = {k: all_f[k] for k in photos}
    return f

train_desc = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)

def dict_to_list(desc):
    all_d = []
    for key in desc.keys():
        [all_d.append(d) for d in desc[key]]
    return all_d

# creating tokenizer class
# this will vectorise text corpus
# each integer will represent token in dictionary

from keras.preprocessing.text import Tokenizer

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


# give each word a index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1


def max_length(desc):
    desc_l = dict_to_list(desc)
    k= max(len(des.split()) for des in desc_l)#finding the max length
    return k

max_l = max_length(descriptions)

def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            # retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list,
                                                                        feature)
            yield [[input_image, input_sequence], output_word]


def create_sequences(tokenizer, max_l, desc_l, f):
    x1, x2, y = list(), list(), list()
    for desc in desc_l:
        s = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(s)):
            out_s, in_s = s[i], s[:i]
            in_s = pad_sequences([in_s], maxlen=max_l)[0]
            out_s = to_categorical([out_s], num_classes=vocab_size)[0]
            x1.append(f)
            x2.append(in_s)
            y.append(out_s)
    return np.array(x1), np.array(x2), np.array(y)


[a, b], c = next(data_generator(train_desc, features, tokenizer, max_length))


# THE CAPTION MODEL
def define_model(vocab_size, max_l):
    #taking the image as 2048 length array
    input_1 = Input(shape=(2048,))
    # dropout for reducing overfitting
    feature_1 = Dropout(0.5)(input_1)
    fe2 = Dense(256, activation='relu')(feature_1)
    # 2nd layer for the model
    input_2 = Input(shape=(max_l,))
    sec_l_1 = Embedding(vocab_size, 256, mask_zero=True)(input_2)
    #dropout for reducing overfitting
    sec_l_2 = Dropout(0.5)(sec_l_1)
    #using in-built LSTM to make se3
    sec_l_3 = LSTM(256)(sec_l_2)
    decoder1 = add([fe2, sec_l_3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[input_1, input_2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print(model.summary())

    return model



print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_desc))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_l)

model = define_model(vocab_size, max_l)
epochs = 10
steps = len(train_desc)
# making a directory models to save our models
os.mkdir("models")
for i in range(25):
    gen = data_generator(train_desc, train_features, tokenizer, max_l)
    model.fit_generator(gen, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")

