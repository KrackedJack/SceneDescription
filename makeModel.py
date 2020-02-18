import numpy as np
from pickle import load, dump
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Embedding
from keras.callbacks import ModelCheckpoint

latent_dims = 256

caps = dict()

with open('captions.pkl', 'rb') as captions:
    caps = load(captions)
print("[ INFO ][ Captions loaded ][ Length: {} ]".format(len(caps)))

features= dict()

with open('extractedfeats.pkl', 'rb') as feats:
    features = load(feats)
print("[ INFO ][ Features loaded ][ Length: {} ]".format(len(features)))

lst = list()
for v in caps.values():
    [lst.append(words) for words in v]

maxLen = max([len(words.split()) for words in lst]) # longest caption

tknzr = Tokenizer()
tknzr.fit_on_texts(lst)
vocab_size = len(tknzr.word_index) + 1

encoder_in, decoder_in, decoder_out = list(), list(), list()
for img, sents in caps.items():
    for cap in sents:
        seq = tknzr.sequences_to_texts([cap])[0]
        for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=maxLen)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                encoder_in.append(features[img])
                decoder_in.append(in_seq)
                decoder_out.append(out_seq)

inputs1 = Input(shape=(None, 25088))
en1 = Dropout(0.3)(inputs1)
en2 = Dense(latent_dims, activation='relu')(en1)
encoder_outputs, state_h, state_c = LSTM(latent_dims, return_state=True)(en1)
encoder_states = [state_h, state_c]

inputs2 = Input(shape=(maxLen,))
de1 = Embedding(vocab_size, latent_dims, mask_zero=True)(inputs2)
de2 = Dropout(0.3)(de1)
decoder = LSTM(latent_dims)(de2, initial_state=encoder_states)
decoder_outputs = Dense(maxLen, activation='softmax')(decoder)

model = Model([inputs1, inputs2], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

print(model.summary())

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

X1train, X1test = encoder_in[:22250], encoder_in[22250:]
X2train, X2test = decoder_in[:22250], decoder_in[22250:]
ytrain, ytest = decoder_out[:22250], decoder_out[22250:]

model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

######## TODO ########
# add inference loop
# measure model performance
# make provision for video input
# add Attention Mechanism
# Expose as an API