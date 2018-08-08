###################
# Date: July 25th 2018
# Sardar
# Description: Parsing the tweets of PHEME dataset for the rumors and not rumors and genrated the output az pheme.csv
# This code reads the json files from 5 different directories of rumors and extracts the text from them
# the main pupose of this pipeline is to get the raw txt of rumor and not rumor in order to be investigated in
# Rumor detection and Belief investigation pipeline
# In this code we load the model by passing the param, the information about how to used the model is in end2end.py description
###################

import json
import os

import preprocessor as p

from home.beliefEng.end2end import load_model,tag


# for rumor in os.path.isdir("NewSeqTAgging/data/test/Rumor/pheme-rnr-dataset"):
def modelLoader():
    param = {'model_name': 84, 'embed_mode': 'prp', 'lrmdethod': 'rmsprop', 'pembed': True, 'epochs': 50, 'crf': True,
             'useNN': True, 'train_embeding': True, 'char': True, 'weights_task': [1.0, 1.0, 1.0],
             'weightgrn_lb': [1.0, 1.0],
             'weights_prevlable': [1.0, 1.0],
             'weights_cblable': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'dir_2_tag': 'rumor',
             'tmode': 'tag'}
    model = load_model(param)
    # tknzr = TweetTokenizer()

    return model

# lcb_tagger("salama halet chetore")










