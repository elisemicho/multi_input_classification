import os
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

CHAR_ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n "
char_start_tag_idx         = len(CHAR_ALPHABETS) + 0
char_end_tag_idx           = len(CHAR_ALPHABETS) + 1
char_unknown_tag_idx       = len(CHAR_ALPHABETS) + 2
char_sent_start_tag_idx    = len(CHAR_ALPHABETS) + 3
char_sent_end_tag_idx      = len(CHAR_ALPHABETS) + 4

CHAR_ALPHABETS_LEN = len(CHAR_ALPHABETS) + 4

labels = ["EGY", "GLF", "LAV", "MSA", "NOR"]
NUM_CLASSES = len(labels)

class Dataset(object):

   def __init__(self, data_frame, vocab_idx, vocab_phone_CZidx, vocab_phone_ENidx, vocab_phone_HUidx, vocab_phone_RUidx):
      
      self.data_frame = data_frame

      # Word vocabulary 
      self.vocab_idx = vocab_idx
      self.vocab_size = len(vocab_idx)

      self.sentence_start_tag_idx = self.vocab_idx["<SOSent>"]
      self.sentence_end_tag_idx   = self.vocab_idx["<EOSent>"]
      self.word_unknown_tag_idx   = self.vocab_idx["<UNK>"]

      # Phone vocabulary
      self.vocab_phone_CZidx = vocab_phone_CZidx
      self.vocab_phone_ENidx = vocab_phone_ENidx
      self.vocab_phone_HUidx = vocab_phone_HUidx
      self.vocab_phone_RUidx = vocab_phone_RUidx


      self.default_unit_dict = {
         "sent_unit"       : "chars",
         "sent_cntx_dir"   : "forward"
      }


   def convertSent2WordIds(self, sentence):
      """
      sentence is a list of word.
      It is converted to list of ids based on vocab_idx
      """

      sent2id = []

      try:
         sent2id = sent2id + [self.vocab_idx[word] if self.vocab_idx[word]<self.vocab_size else self.word_unknown_tag_idx for word in sentence]
      except KeyError as e:
         print(e)
         print (sentence)
         raise ValueError('Fix this issue')

      return sent2id

   def convertSent2CharIds(self, sentence):
      """
      sentence is a string.
      It is converted to list of ids based on CHARACTERS_ALPHABET
      """

      sent2id = []

      try:
         sent2id = sent2id + [CHAR_ALPHABETS.find(char) for char in sentence]
      except KeyError as e:
         print(e)
         print (sentence)
         raise ValueError('Fix this issue')

      return sent2id


   def convertSeq2PhoneIds(self, sequence, phone_idx):
      """
      sequence is a list of phones
      It is converted to list of ids based on phone_idx
      """

      sent2id = []

      try:
         sent2id = sent2id + [phone_idx[word] for word in sequence]
      except KeyError as e:
         print(e)
         print (sequence)
         raise ValueError('Fix this issue')

      return sent2id


   def generate(self, unit_dict=None, has_class=True, add_start_end_tag=False):
      """
      unit_dict contains these keys that can have
      sent_unit       can be ["words", chars"]
      sent_cntx_dir   can be ["forward", "backward"]

      """
      if not unit_dict:
         unit_dict = self.default_unit_dict

      try:
         unit_dict["sent_cntx_dir"]
      except KeyError as e:
         unit_dict["sent_cntx_dir"] = "forward"

      self.ids_sentence   = []
      self.ids_label      = []
      self.ids_phones_CZ  = []
      self.ids_phones_EN  = []
      self.ids_phones_HU  = []
      self.ids_phones_RU  = []
      self.ids_embed      = []

      for index in self.data_frame.index:

         # sentence --------------------------------------------------------------
         if unit_dict["sent_unit"] == "words":

            # if unit_dict["sent_unit"] == "words":
            text_word_list = [word_id for word_id in self.convertSent2WordIds(self.data_frame.Words[index])]

            if unit_dict["sent_cntx_dir"] == "backward":
               text_word_list = text_word_list[::-1]

            self.ids_sentence.append(text_word_list)

         elif unit_dict["sent_unit"] == "chars":

            text_char_list = [char_id for char_id in self.convertSent2CharIds(self.data_frame.Chars[index])]

            self.ids_sentence.append(text_char_list)

         else:
            assert False, "give valid sent_unit key-value"

         # label --------------------------------------------------------------
         if has_class:
            self.ids_label.append(labels.index(self.data_frame.Class[index]))
         
         # phones --------------------------------------------------------------
         self.ids_phones_CZ.append(self.convertSeq2PhoneIds(self.data_frame.Phone_CZ[index], self.vocab_phone_CZidx))
         self.ids_phones_EN.append(self.convertSeq2PhoneIds(self.data_frame.Phone_EN[index], self.vocab_phone_ENidx))
         self.ids_phones_HU.append(self.convertSeq2PhoneIds(self.data_frame.Phone_HU[index], self.vocab_phone_HUidx))
         self.ids_phones_RU.append(self.convertSeq2PhoneIds(self.data_frame.Phone_RU[index], self.vocab_phone_RUidx))

         # acoustic --------------------------------------------------------------
         self.ids_embed.append(self.data_frame.Embed[index])


   def preprocess(self, MAX_CHAR_IN_SENT_LEN, MAX_PHONE_CZ_LEN, MAX_PHONE_EN_LEN, MAX_PHONE_HU_LEN, MAX_PHONE_RU_LEN):

      print('Padding character sequences')

      self.ids_sentence = pad_sequences(self.ids_sentence, maxlen=MAX_CHAR_IN_SENT_LEN, value=char_unknown_tag_idx,
                                        padding="post",truncating="post")

      print(self.ids_sentence.shape)


      print('Padding phone sequences')

      self.ids_phones_CZ = pad_sequences(self.ids_phones_CZ, maxlen=MAX_PHONE_CZ_LEN, value=len(self.vocab_phone_CZidx))
      self.ids_phones_EN = pad_sequences(self.ids_phones_EN, maxlen=MAX_PHONE_EN_LEN, value=len(self.vocab_phone_ENidx))
      self.ids_phones_HU = pad_sequences(self.ids_phones_HU, maxlen=MAX_PHONE_HU_LEN, value=len(self.vocab_phone_HUidx))
      self.ids_phones_RU = pad_sequences(self.ids_phones_RU, maxlen=MAX_PHONE_RU_LEN, value=len(self.vocab_phone_RUidx))

      print(self.ids_phones_CZ.shape, self.ids_phones_EN.shape, self.ids_phones_HU.shape, self.ids_phones_RU.shape)


      print('Turning labels in one-hot vectors')

      self.ids_label = np.array(self.ids_label)
      self.ids_label = to_categorical(np.array(self.ids_label), NUM_CLASSES)
      print(self.ids_label.shape)


      print('Taking ready-made acoustic embeddings')

      self.ids_embed = np.array(self.ids_embed)
      print(self.ids_embed.shape)

def safe_mkdir(path):
   """ Create a directory if there isn't one already. """
   try:
     os.mkdir(path)
   except OSError:
     pass