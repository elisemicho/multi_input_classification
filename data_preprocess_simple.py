''' Data preparation '''
# coding: utf-8

# Setup

import pandas as pd
import itertools
import pickle

from config import DataConfig

# Load data

data_config = DataConfig()

train_df = pd.read_csv(data_config.data_source+"/train.all", sep="\t", engine='python', 
                            header=None, skiprows=0, names=["Class","Text","Phone_CZ", "Phone_EN", "Phone_HU", "Phone_RU", "Embed"])
test_df = pd.read_csv(data_config.data_source+"/dev.all", sep="\t", engine='python', 
                           header=None, skiprows=0, names=["Class","Text","Phone_CZ", "Phone_EN", "Phone_HU", "Phone_RU", "Embed"])
train_df = train_df.dropna()
test_df = test_df.dropna()

print("Train and Test shape : ",train_df.shape, test_df.shape)

# Pre process data frames

train_df['Chars'] = train_df.Text.apply(lambda s: s)
test_df['Chars'] = test_df.Text.apply(lambda s: s)
train_df['Words'] = train_df.Text.apply(lambda s: s.split())
test_df['Words'] = test_df.Text.apply(lambda s: s.split())

train_df.Phone_CZ = train_df.Phone_CZ.apply(lambda p: p.split())
test_df.Phone_CZ = test_df.Phone_CZ.apply(lambda p: p.split())
train_df.Phone_EN = train_df.Phone_EN.apply(lambda p: p.split())
test_df.Phone_EN = test_df.Phone_EN.apply(lambda p: p.split())
train_df.Phone_HU = train_df.Phone_HU.apply(lambda p: p.split())
test_df.Phone_HU = test_df.Phone_HU.apply(lambda p: p.split())
train_df.Phone_RU = train_df.Phone_RU.apply(lambda p: p.split())
test_df.Phone_RU = test_df.Phone_RU.apply(lambda p: p.split())

train_df.Embed = train_df.Embed.apply(lambda p: p.split())
test_df.Embed = test_df.Embed.apply(lambda p: p.split())

train_df.drop(["Text"], axis=1, inplace=True)
test_df.drop(["Text"], axis=1, inplace=True)

# Save the pandas processed frame

store = pd.HDFStore(data_config.data_source+'/dataset.h5')
store['train_df'] = train_df
store['test_df'] = test_df
store.close()

# Generate vocabularies

# ### words

train_words = train_df.Words
train_words = list(itertools.chain.from_iterable(train_words))
test_words = test_df.Words
test_words = list(itertools.chain.from_iterable(test_words))
vocab_words = list(set(train_words) | set(test_words))

# add extra words such as start/end of sentence
vocab_words.append("<UNK>")
vocab_words.append("<SOSent>")
vocab_words.append("<EOSent>")
vocab_words.append("<SODoc>")
vocab_words.append("<EODoc>")
vocab_wordidx = {w:i for i,w in enumerate(vocab_words)}

with open(data_config.data_source+'/vocab_words_wordidx.pkl', 'wb') as f:
    pickle.dump((vocab_words, vocab_wordidx), f, protocol=pickle.HIGHEST_PROTOCOL)

# ### Phones

train_phone_CZ = list(itertools.chain.from_iterable(train_df.Phone_CZ))
test_phone_CZ = list(itertools.chain.from_iterable(test_df.Phone_CZ))
vocab_phone_CZ = list(set(train_phone_CZ) | set(test_phone_CZ))
vocab_phone_CZidx = {w:i for i,w in enumerate(vocab_phone_CZ)}
with open(data_config.data_source+'/vocab_phone_CZ_phone_CZidx.pkl', 'wb') as f:
    pickle.dump((vocab_phone_CZ, vocab_phone_CZidx), f, protocol=pickle.HIGHEST_PROTOCOL)

train_phone_EN = list(itertools.chain.from_iterable(train_df.Phone_EN))
test_phone_EN = list(itertools.chain.from_iterable(test_df.Phone_EN))
vocab_phone_EN = list(set(train_phone_EN) | set(test_phone_EN))
vocab_phone_ENidx = {w:i for i,w in enumerate(vocab_phone_EN)}
with open(data_config.data_source+'/vocab_phone_EN_phone_ENidx.pkl', 'wb') as f:
    pickle.dump((vocab_phone_EN, vocab_phone_ENidx), f, protocol=pickle.HIGHEST_PROTOCOL)

train_phone_HU = list(itertools.chain.from_iterable(train_df.Phone_HU))
test_phone_HU = list(itertools.chain.from_iterable(test_df.Phone_HU))
vocab_phone_HU = list(set(train_phone_HU) | set(test_phone_HU))
vocab_phone_HUidx = {w:i for i,w in enumerate(vocab_phone_HU)}
with open(data_config.data_source+'/vocab_phone_HU_phone_HUidx.pkl', 'wb') as f:
    pickle.dump((vocab_phone_HU, vocab_phone_HUidx), f, protocol=pickle.HIGHEST_PROTOCOL)

train_phone_RU = list(itertools.chain.from_iterable(train_df.Phone_RU))
test_phone_RU = list(itertools.chain.from_iterable(test_df.Phone_RU))
vocab_phone_RU = list(set(train_phone_RU) | set(test_phone_RU))
vocab_phone_RUidx = {w:i for i,w in enumerate(vocab_phone_RU)}
with open(data_config.data_source+'/vocab_phone_RU_phone_RUidx.pkl', 'wb') as f:
    pickle.dump((vocab_phone_RU, vocab_phone_RUidx), f, protocol=pickle.HIGHEST_PROTOCOL)
