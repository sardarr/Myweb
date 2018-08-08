import os
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)

        self.vocab_cbtags  = load_vocab(self.filename_cbtags)
        self.vocab_prptags  = load_vocab(self.filename_prptags)

        self.vocab_gentags  = load_vocab(self.filename_gentags)

        self.vocab_prp_chars = load_vocab(self.filename_prp_chars)

        self.vocab_tag_ind={'0': 'O', '1': 'rob', '2': 'na', '3': 'cb', '4': 'ncb'}
        self.nwords     = len(self.vocab_words)

        self.nprpchars     = len(self.vocab_prp_chars)

        self.ncbtags      = len(self.vocab_cbtags)
        self.nprptags      = len(self.vocab_prptags)
        self.ngentags      = len(self.vocab_gentags)

        self.aug=False

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_prp_chars, lowercase=True, chars=self.use_chars)

        self.processing_cbtag  = get_processing_word(self.vocab_cbtags,
                lowercase=False, allow_unk=False)
        self.processing_prptag  = get_processing_word(self.vocab_prptags,
                lowercase=False, allow_unk=False)


        self.processing_gentags  = get_processing_word(self.vocab_gentags,
                lowercase=False, allow_unk=False)
        # 3. get pre-trained embeddings

        self.prp_embeddings = (get_trimmed_glove_vectors(self.filename_prp_trimmed)
                if self.use_pretrained else None)

    # general config
    dir_output = "log/"
    # dir_model  = "results/crf_50/model.weights/"
    dir_model  = "home/beliefEng/model/lcbmodel/"
    pr_tag=dir_model+'incorect_tags/'
    path_log   = dir_output + "log.txt"
    ####embedding Mode should change by thee prp or ev
    embed_mode="None"
    model_mode=['None']
    # embeddings
    fp_log=True
    dim_word = 100
    dim_char = 100

    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_prp_trimmed = "home/beliefEng/model/lcbmodel/data_prp/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True
    trimmed_mix_filename = "data/mix/glove.6B.{}d.trimmed.npz".format(dim_word)

    # dataset

    filename_second_train = "data/transfer.txt"
    filename_test = "home/beliefEng/model/lcbmodel/data_prp/14test.txt"
    filename_train = "home/beliefEng/model/lcbmodel/data_prp/14train.txt"
    filename_t2train="data/data_ev/15to17.txt"
    filename_dev = "data/data_ev/17test.txt"
    train_mtl="data/data_ev16/151617WS_MTL.txt"
    test_mtl="data/data_ev16/16test_MTL.txt"
    test_mtl_df="data/data_ev_df/16test_MTL.txt"
    test_mtl_nw="data/data_ev_ny/16test_MTL.txt"

    mix_mtl = "data/mix/mix.txt"

    filename_ev_trimmed = "data/data_ev16/glove.6B.{}d.trimmed.npz".format(dim_word)

    year=filename_train.split("/")[1].split(".")[0]+"/"
    # filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "home/beliefEng/model/lcbmodel/data_prp/words.txt"
    filename_cbtags = "home/beliefEng/model/lcbmodel/data_prp/cbtags.txt"
    filename_prptags = "home/beliefEng/model/lcbmodel/data_prp/prptags.txt"
    filename_evtags = "data/data_ev_df/evtags.txt"
    filename_gentags = "home/beliefEng/model/lcbmodel/data_prp/gentags.txt"

    filename_ev_words = "data/data_ev16/words.txt"
    filename_prp_chars = "home/beliefEng/model/lcbmodel/data_prp/chars.txt"
    filename_ev_chars = "data/data_ev16/chars.txt"

    words_mix_filename = "data/mix/words.txt"
    tags_mix_filename = "data/mix/evtags.txt"
    chars_mix_filename = "data/mix/chars.txt"

    tagpath="data/test/"
    # training
    train_embeddings = False
    nepochs          = 100
    dropout          = 0.5
    weight=1
    batch_size       = 20
    lr_method        = "RMSProp"
    lr               = 0.003
    lr_decay         = 0.8
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3
    weights_cblable=[1,1,1,1,1,1,1,1,1,1,1,1,1]
    weights_prevlable = [1,1]
    weightgrn_lb = [1,1]
    weights_task= [1, 1, 1]
    model_name=""
    model_2_restore="84/"
    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_doc= True #this is just for the doc embading for the
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU
    usNN=True
