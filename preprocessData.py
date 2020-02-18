import pandas as pd
import numpy as np 
import sys
import pickle as pkl

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

def img_data(name):
    img_path = 'flickr30k_images/flickr30k_images'
    img = image.load_img(img_path + name, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    return preprocess_input(img_data)

caps = pd.read_csv("flickr30k_images/results.csv", delimiter="|")

model = VGG16(weights="imagenet", include_top=False)

feats = dict()
i = 1
for name in caps["image_name"].unique():
    feats[name] = np.array(model.predict(img_data(name))).flatten()
print("[ INFO ][ Features Extracted ][ Length: {} ]".format(len(feats)))

pkl.dump("extractedFeatures.pkl", feats)
del feats 

captions = dict()
for row in caps.itertuples():
    temp = row["comment"].split()
    temp = [word.lower().replace("", str.punctuation) for word in temp if len(word)>1]
    temp = "startseq " + " ".join(temp) + " endseq" 

    if row["image_name"] not in captions:
        captions["image_name"] = [temp]
    else:
        captions["image_name"].append(temp)

print("[ INFO ][ Captions Parsed ][ Length: {} ]".format(len(captions)))
pkl.dump("captions.pkl", captions)