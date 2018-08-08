################
#July 26
#Sardar
#This code is to use the belief system as end to end pipeline
#It could be read by other codes or to be run as single pipelie for belief tagging
# This containes the methods required for Belief end to end code.
# How to Use:
#   1- using the Param from the Param.txt after: import end2end   param={'Model name':....,....}
#   2- Import the load_model method from this code by passing the param Exp. from model=load_model(param)
#   3- calling the tag method and passing a string and model. Tag method returns the tagged string in return Exp. tag(model, "Your STR")
#################
from nltk import TweetTokenizer

from home.beliefEng.model.MTLModel import MTLModel
from home.beliefEng.model.config import Config


def tag(model,token_list):
    """Creates interactive shell to play with model
    """
    # token_list =sentence.strip().split(" ")
    predLcb,predEvPrp,predGen = model.predict(token_list)
    cb_tagged=""
    for tInd in range(len(token_list)):
        if predLcb[tInd] == "O":
            cb_tagged+=token_list[tInd]+" "
        else:
            cb_tagged+="<"+predLcb[tInd]+">"+token_list[tInd]+"</"+predLcb[tInd]+"> "
    return cb_tagged

def tag_fp(pred,sentence):
    """
    This method is being used in False positive logging by getting the list of input words as a list
    and coressponding cb labels in a list
        # pred=['O', 'O', 'cb', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    # sentence=['You', 'also', 'left', 'out', 'Rush', ',', 'Hannity', 'and', 'Lavin', '.']

    output = You also <cb>left</cb>...]
    """

    cb_tagged=""
    for tInd in range(len(sentence)):
        if pred[tInd] == "O":
            cb_tagged+=sentence[tInd]+" "
        else:
            cb_tagged+="<"+pred[tInd]+">"+sentence[tInd]+"</"+pred[tInd]+"> "
    return cb_tagged






def load_model(param):
    # create instance of config
    config = Config()
    config.weights_cblable=param['weights_cblable']
    config.weights_prevlable=param['weights_prevlable']
    config.model_name=param['model_name']
    config.weightgrn_lb = param['weightgrn_lb']
    config.weights_task = param['weights_task']
    config.embed_mode = param['embed_mode']
    config.embeddings=param['embed_mode']
    config.nepochs=param['epochs']

    config.lr_method=param['lrmdethod']
    if param['pembed']==None:
        config.filename_prp_trimmed=param['pembed']
    config.use_crf=param['crf']
    config.use_chars=param['char']
    config.train_embeddings=param['train_embeding']
    config.usNN=param['useNN']

    model = MTLModel(config)
    model.trfbuild()
    model.restore_session()
    return model


