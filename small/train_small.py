#the dataset can be downloaded at https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip and https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
#then the datasets need to be named according to the programme

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

def text_vocabulary(desc):
    vocab = set()
    for key in desc.keys():
        [vocab.update(d.split()) for d in desc[key]]
     return vocab


def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_c in enumerate(caps):
            img_c.replace("-", " ")
            d = img_c.split()
            d = [word.lower() for word in d]
            d = [word.translate(table) for word in d]
            d = [word for word in d if (word.isalpha())]
            d = [word for word in d if (len(word) >=2)]
            img_c = ' '.join(d)
            captions[img][i] = img_c
    return captions

def extract_features(directory):
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)

        image = image / 127.5
        image = image - 1.0

        feature = model.predict(image)
        features[img] = feature
    return features

# load the data
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos

def dict_to_list(desc):
    all_d = []
    for key in desc.keys():
        [all_d.append(d) for d in desc[key]]
    return all_d

def load_features(imgs):
    all_features = load(open("features.p", "rb"))
    f = {i: all_features[i] for i in imgs}
    return f


def load_clean_descriptions(filename, photos):
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





def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


from keras.preprocessing.text import Tokenizer


def max_length(desc):
    desc_l = dict_to_list(desc)
    k= max(len(des.split()) for des in desc_l)#finding the max length
    return k

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

def create_sequences(tokenizer, max_l, desc_l, f):
    x1, x2, y = list(), list(), list()#initialising as empty lists
    for desc in desc_l:
        s = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(s)):
            out_s, in_s = s[i], s[:i]
            #getting the out_s
            out_s = to_categorical([out_s], num_classes=vocab_size)[0]
            #getting the in_s
            in_s = pad_sequences([in_s], maxlen=max_l)[0]
            x1.append(f)
            y.append(out_s)
            x2.append(in_s)
    a,b,c=np.array(x1), np.array(x2), np.array(y)
    return a,b,c


def data_generator(desc, fs, tokenizer, max_l):
    while True:
        for key, desc_list in desc.items():
            f = fs[key][0]
            in_img, in_sequence, out_w = create_sequences(tokenizer, max_l, desc_list,f)
            #creating a generator object
            yield [[in_img, in_sequence], out_w]








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

dataset_text = "small_text"
dataset_images = "small_dataset/photos"

filename = dataset_text + "/" + "Flickr8k.token.txt"
descriptions = all_img_captions(filename)
print("Length of descriptions =", len(descriptions))
clean_descriptions = cleaning_text(descriptions)
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
# saving each description to file
save_descriptions(clean_descriptions, "descriptions.txt")
# 2048 feature vector
features = extract_features(dataset_images)
dump(features, open("features.p", "wb"))
features = load(open("features.p", "rb"))

fname = dataset_text + "/" + "Flickr_8k.trainImages.txt"
train_images = load_photos(fname)
train_desc = load_clean_descriptions("descriptions.txt", train_images)
train_features = load_features(train_images)
tokenizer = create_tokenizer(train_desc)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
max_l = max_length(descriptions)
# train our model
print('Dataset: ', len(train_images))
print('Descriptions: train=', len(train_desc))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_l)

model = define_model(vocab_size, max_length)

epochs = 10
steps = len(train_descriptions)

for i in range(0, 26):
    gen = data_generator(train_descriptions, train_features, tokenizer, max_l)
    model.fit_generator(gen, epochs=1, steps_per_epoch=steps, verbose=1)
    #saving the model
    model.save("models/model_" + str(i) + ".h5")
