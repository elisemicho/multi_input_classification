import pandas as pd
import pickle

# from models.multi_input_char_cnn import MultiInputCNN
# from models.multi_input_with_dropout import MultiInputCNN
# from models.multi_input_concat_before_conv import MultiInputCNN
from models.multi_input_char_acoustic import MultiInputCNN

import dataset_utils

from config import DataConfig
from config import MultiInputCNNConfig
from config import TrainingConfig

# import tensorflow as tf
# tf.flags.DEFINE_string("model", "zhang", "Specifies which model to use. (default: 'zhang')")
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

def load_data(data_source):

    print('Loading data')

    store = pd.HDFStore(data_source+'/dataset.h5')
    train_df = store['train_df']
    val_df = store['test_df']

    print(train_df.shape)
    print(val_df.shape)

    print(train_df.Class.value_counts())
    print(val_df.Class.value_counts())

    return train_df, val_df

def load_vocabularies(data_source):

    print('Loading vocabularies')

    print('Words')
    corpus_vocab_list, corpus_vocab_wordidx = None, None
    with open(data_source+'/vocab_words_wordidx.pkl', 'rb') as f:
      (corpus_vocab_list, corpus_wordidx) = pickle.load(f)
    print(len(corpus_vocab_list), len(corpus_wordidx))
    vocab_idx = corpus_wordidx

    print('Phones')
    corpus_vocab_phone_CZ, corpus_phone_CZidx = None, None
    with open(data_source+'/vocab_phone_CZ_phone_CZidx.pkl', 'rb') as f:
      (corpus_vocab_phone_CZ, corpus_phone_CZidx) = pickle.load(f)
    print(len(corpus_vocab_phone_CZ), len(corpus_phone_CZidx))
    vocab_phone_CZidx = corpus_phone_CZidx

    corpus_vocab_phone_EN, corpus_phone_ENidx = None, None
    with open(data_source+'/vocab_phone_EN_phone_ENidx.pkl', 'rb') as f:
      (corpus_vocab_phone_EN, corpus_phone_ENidx) = pickle.load(f)
    print(len(corpus_vocab_phone_EN), len(corpus_phone_ENidx))
    vocab_phone_ENidx = corpus_phone_ENidx

    corpus_vocab_phone_HU, corpus_phone_HUidx = None, None
    with open(data_source+'/vocab_phone_HU_phone_HUidx.pkl', 'rb') as f:
      (corpus_vocab_phone_HU, corpus_phone_HUidx) = pickle.load(f)
    print(len(corpus_vocab_phone_HU), len(corpus_phone_HUidx))
    vocab_phone_HUidx = corpus_phone_HUidx

    corpus_vocab_phone_RU, corpus_phone_RUidx = None, None
    with open(data_source+'/vocab_phone_RU_phone_RUidx.pkl', 'rb') as f:
      (corpus_vocab_phone_RU, corpus_phone_RUidx) = pickle.load(f)
    print(len(corpus_vocab_phone_RU), len(corpus_phone_RUidx))  
    vocab_phone_RUidx = corpus_phone_RUidx

    return vocab_idx, vocab_phone_CZidx, vocab_phone_ENidx, vocab_phone_HUidx, vocab_phone_RUidx

def find_max_len(x_train, x_val):

    max_len = 0
    for item in x_train:
       current_len = len(item)
       if current_len > max_len:
           max_len = current_len
    for item in x_val:
       current_len = len(item)
       if current_len > max_len:
           max_len = current_len
    return max_len


if __name__ == "__main__":
    
    print('Loading data')

    data_config = DataConfig()

    train_df, val_df = load_data(data_config.data_source)
    vocab_idx, vocab_phone_CZidx, vocab_phone_ENidx, vocab_phone_HUidx, vocab_phone_RUidx = load_vocabularies(data_config.data_source)

    # Get useful dimensions from vocabularies
    PHONE_CZ_ALPHABET_LEN = len(vocab_phone_CZidx)+1
    PHONE_EN_ALPHABET_LEN = len(vocab_phone_ENidx)+1
    PHONE_HU_ALPHABET_LEN = len(vocab_phone_HUidx)+1
    PHONE_RU_ALPHABET_LEN = len(vocab_phone_RUidx)+1


    print('Generating ids')

    custom_unit_dict = {"sent_unit": "chars"}

    training_data = dataset_utils.Dataset(train_df, vocab_idx, vocab_phone_CZidx, vocab_phone_ENidx, vocab_phone_HUidx, vocab_phone_RUidx)
    training_data.generate(custom_unit_dict, has_class=True, add_start_end_tag=False)

    validation_data = dataset_utils.Dataset(val_df, vocab_idx, vocab_phone_CZidx, vocab_phone_ENidx, vocab_phone_HUidx, vocab_phone_RUidx)
    validation_data.generate(custom_unit_dict, has_class=True, add_start_end_tag=False)

    # Get useful dimensions from dataset
    MAX_CHAR_IN_SENT_LEN = find_max_len(training_data.ids_sentence, validation_data.ids_sentence)
    MAX_PHONE_CZ_LEN = find_max_len(training_data.ids_phones_CZ, validation_data.ids_phones_CZ)
    MAX_PHONE_EN_LEN = find_max_len(training_data.ids_phones_EN, validation_data.ids_phones_EN)
    MAX_PHONE_HU_LEN = find_max_len(training_data.ids_phones_HU, validation_data.ids_phones_HU)
    MAX_PHONE_RU_LEN = find_max_len(training_data.ids_phones_RU, validation_data.ids_phones_RU)


    print('Preprocessing data')

    training_data.preprocess(MAX_CHAR_IN_SENT_LEN, MAX_PHONE_CZ_LEN, MAX_PHONE_EN_LEN, MAX_PHONE_HU_LEN, MAX_PHONE_RU_LEN)
    validation_data.preprocess(MAX_CHAR_IN_SENT_LEN, MAX_PHONE_CZ_LEN, MAX_PHONE_EN_LEN, MAX_PHONE_HU_LEN, MAX_PHONE_RU_LEN)

    ACOUSTIC_EMB_SIZE = training_data.ids_embed.shape[1]

    training_data = ({'sent_input': training_data.ids_sentence, 'phone_CZ_input': training_data.ids_phones_CZ, 
                     'phone_EN_input': training_data.ids_phones_EN, 'phone_HU_input': training_data.ids_phones_HU, 
                     'phone_RU_input': training_data.ids_phones_RU, 'embed_input': training_data.ids_embed},
                      {'l_out': training_data.ids_label})    

    validation_data=({'sent_input': validation_data.ids_sentence, 'phone_CZ_input': validation_data.ids_phones_CZ, 
                     'phone_EN_input': validation_data.ids_phones_EN, 'phone_HU_input': validation_data.ids_phones_HU, 
                     'phone_RU_input': validation_data.ids_phones_RU, 'embed_input': validation_data.ids_embed},
                      {'l_out': validation_data.ids_label})

    training_inputs, training_labels = training_data
    validation_inputs, validation_labels = validation_data

    # Load model configurations
    # if FLAGS.model == "zhang":
    model_config = MultiInputCNNConfig()
    # Build model
    model = MultiInputCNN(input_char_size=MAX_CHAR_IN_SENT_LEN,
                          input_phone_CZ_size=MAX_PHONE_CZ_LEN,
                          input_phone_EN_size=MAX_PHONE_EN_LEN,
                          input_phone_HU_size=MAX_PHONE_HU_LEN,
                          input_phone_RU_size=MAX_PHONE_RU_LEN,
                          input_acoustic_size=ACOUSTIC_EMB_SIZE,
                          phone_CZ_alphabet_size=PHONE_CZ_ALPHABET_LEN,
                          phone_EN_alphabet_size=PHONE_EN_ALPHABET_LEN,
                          phone_HU_alphabet_size=PHONE_HU_ALPHABET_LEN,
                          phone_RU_alphabet_size=PHONE_RU_ALPHABET_LEN,
                          model_config=model_config,
                          data_config=data_config)

    # Load training configurations
    training_config = TrainingConfig()
    # Train model
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=training_config.epochs,
                batch_size=training_config.batch_size,
                checkpoint_every=training_config.checkpoint_every)
    # Test model
    model.test(testing_inputs=validation_inputs,
               testing_labels=validation_labels)