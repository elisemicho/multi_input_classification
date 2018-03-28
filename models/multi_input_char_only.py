
# coding: utf-8

# # Setup

import tensorflow.contrib.keras as keras
import tensorflow as tf

from keras import backend as K

from keras.engine import Layer, InputSpec, InputLayer

from keras.models import Model, Sequential

from keras.layers import Dropout, Embedding, concatenate
from keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPool2D, ZeroPadding1D, GlobalMaxPool1D
from keras.layers import Dense, Input, Flatten, BatchNormalization
from keras.layers import Concatenate, Dot, Merge, Multiply, RepeatVector
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import SimpleRNN, LSTM, GRU, Lambda, Permute

from keras.layers.core import Reshape, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.constraints import maxnorm
from keras.regularizers import l2

import dataset_utils
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


class MultiInputCNN(object):

    def __init__(self, input_char_size, input_phone_CZ_size, input_phone_EN_size, input_phone_HU_size, input_phone_RU_size, input_acoustic_size,
                phone_CZ_alphabet_size, phone_EN_alphabet_size, phone_HU_alphabet_size, phone_RU_alphabet_size, model_config, data_config):

        # print('Initializing embedding layers for characters and phones')

        # # ### for characters

        # char_embeddings = np.random.randn(global_dataset_utils_simple.CHAR_ALPHABETS_LEN, CHAR_EMB_SIZE)
        # print(char_embeddings.shape)

        # # for phones

        # phone_CZ_embeddings = np.random.randn(PHONE_CZ_ALPHABET_LEN, PHONE_CZ_EMB_SIZE)
        # print(phone_CZ_embeddings.shape)

        # phone_EN_embeddings = np.random.randn(PHONE_EN_ALPHABET_LEN, PHONE_EN_EMB_SIZE)
        # print(phone_EN_embeddings.shape)

        # phone_HU_embeddings = np.random.randn(PHONE_HU_ALPHABET_LEN, PHONE_HU_EMB_SIZE)
        # print(phone_HU_embeddings.shape)

        # phone_RU_embeddings = np.random.randn(PHONE_RU_ALPHABET_LEN, PHONE_RU_EMB_SIZE)
        # print(phone_RU_embeddings.shape)


        # Input parameters

        self.input_char_size = input_char_size
        self.input_phone_CZ_size = input_phone_CZ_size
        self.input_phone_EN_size = input_phone_EN_size
        self.input_phone_HU_size = input_phone_HU_size
        self.input_phone_RU_size = input_phone_RU_size
        self.input_acoustic_size = input_acoustic_size

        # Embedding parameters

        self.phone_CZ_alphabet_size = phone_CZ_alphabet_size
        self.phone_EN_alphabet_size = phone_EN_alphabet_size
        self.phone_HU_alphabet_size = phone_HU_alphabet_size
        self.phone_RU_alphabet_size = phone_RU_alphabet_size

        self.model_config = model_config
        self.data_config = data_config

        self._build_model()


    def _build_model(self):

        print('Building the model')

        # Embedding layers
        sent_input = Input(shape=(self.input_char_size,), dtype='int32', name='sent_input')
        char_embedding = Embedding(dataset_utils.CHAR_ALPHABETS_LEN, self.model_config.char_emb_size, input_length=self.input_char_size, trainable=True)(sent_input)

        phone_CZ_input = Input(shape=(self.input_phone_CZ_size,), dtype='int32', name='phone_CZ_input')
        #phone_CZ_embedding = Embedding(self.phone_CZ_alphabet_size, self.model_config.phone_CZ_emb_size, input_length=self.input_phone_CZ_size, trainable=True)(phone_CZ_input)

        phone_EN_input = Input(shape=(self.input_phone_EN_size,), dtype='int32', name='phone_EN_input')
        #phone_EN_embedding = Embedding(self.phone_EN_alphabet_size, self.model_config.phone_EN_emb_size, input_length=self.input_phone_EN_size, trainable=True)(phone_EN_input)

        phone_HU_input = Input(shape=(self.input_phone_HU_size,), dtype='int32', name='phone_HU_input')
        #phone_HU_embedding = Embedding(self.phone_HU_alphabet_size, self.model_config.phone_HU_emb_size, input_length=self.input_phone_HU_size, trainable=True)(phone_HU_input)

        phone_RU_input = Input(shape=(self.input_phone_RU_size,), dtype='int32', name='phone_RU_input')
        #phone_RU_embedding = Embedding(self.phone_RU_alphabet_size, self.model_config.phone_RU_emb_size, input_length=self.input_phone_RU_size, trainable=True)(phone_RU_input)

        embed_input = Input(shape=(self.input_acoustic_size,), dtype='float32', name='embed_input')

        # Convolutional layers on chars and phones
        conv_pools = []
        for embedding in [char_embedding]:
            for filter_size in self.model_config.filter_sizes:
                l_zero = ZeroPadding1D((filter_size-1,filter_size-1))(embedding)
                l_conv = Conv1D(filters=self.model_config.num_filters, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
                l_pool = GlobalMaxPool1D()(l_conv)
                conv_pools.append(l_pool)
            
        x = Concatenate(axis=1)(conv_pools)

        # Concatenate with acoustic embeddings
        # x = Concatenate()([l_merge,embed_input])   

        # Fully connected layers 
        for fl in self.model_config.fully_connected_layers:
            x = Dense(fl, activation='relu', kernel_regularizer=l2(0.01))(x)
        
        l_out = Dense(dataset_utils.NUM_CLASSES, activation='softmax', name ="l_out")(x)

        model = Model(inputs=[sent_input, phone_CZ_input, phone_EN_input, phone_HU_input, phone_RU_input, embed_input], outputs=l_out)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_accuracy'])
        self.model = model
        self.model.summary()


    def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels,
              epochs, batch_size, checkpoint_every=100):

        print('Training the model')

        # Create callbacks
        dataset_utils.safe_mkdir('results')
        dataset_utils.safe_mkdir('results/'+self.data_config.name)
        dataset_utils.safe_mkdir('results/'+self.data_config.name+'/'+self.model_config.name)
        dataset_utils.safe_mkdir('results/'+self.data_config.name+'/'+self.model_config.name+'/tb_graphs')
        tb_callback = keras.callbacks.TensorBoard(log_dir='results/'+self.data_config.name+'/'+self.model_config.name+'/tb_graphs/', histogram_freq=checkpoint_every, 
                                                    write_graph=True, write_images=True)

        checkpointer = ModelCheckpoint(filepath='results/'+self.data_config.name+'/'+self.model_config.name+'/model_weights.hdf5', 
                                            verbose=1,
                                            monitor="val_categorical_accuracy",
                                            save_best_only=True,
                                            mode="max")
        earlystopping = EarlyStopping(monitor='val_categorical_accuracy', 
                                      min_delta=0, patience=5, 
                                      verbose=1, mode='auto')
        
        with tf.Session() as sess:
            # model = keras.models.load_model('current_model.h5')
            sess.run(tf.global_variables_initializer())
            try:
                self.model.load_weights('results/'+self.data_config.name+'/'+self.model_config.name+'/model_weights.hdf5')
            except IOError as ioe:
                print("no checkpoints available !")

            self.model.fit(training_inputs, training_labels,
                           validation_data=(validation_inputs, validation_labels),
                            epochs=epochs, batch_size=batch_size, shuffle=True,
                            callbacks=[tb_callback,checkpointer,earlystopping])

            print("\nFinal evaluation\n")

            predictions = self.model.predict(validation_inputs)
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(validation_labels['l_out'], axis=1)

            print("f1_score\n", f1_score(y_true, y_pred, average="macro"))
            print("accuracy_score\n", accuracy_score(y_true, y_pred))
            print("\nclassification_report\n",classification_report(y_true, y_pred, target_names=dataset_utils.labels))
            print("\nconfusion_matrix\n",confusion_matrix(y_true, y_pred))

    def test(self, testing_inputs, testing_labels):

         with tf.Session() as sess:
            # model = keras.models.load_model('current_model.h5')
            sess.run(tf.global_variables_initializer())
            try:
                self.model.load_weights('results/'+self.data_config.name+'/'+self.model_config.name+'/model_weights.hdf5')
            except IOError as ioe:
                print("no checkpoints available !")

            print("\nEvaluation on best model\n")

            predictions = self.model.predict(testing_inputs)
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(testing_labels['l_out'], axis=1)

            print("f1_score\n", f1_score(y_true, y_pred, average="macro"))
            print("accuracy_score\n", accuracy_score(y_true, y_pred))
            print("\nclassification_report\n",classification_report(y_true, y_pred, target_names=dataset_utils.labels))
            print("\nconfusion_matrix\n",confusion_matrix(y_true, y_pred))
   

    
