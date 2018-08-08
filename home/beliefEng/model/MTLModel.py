import numpy as np
import os
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from .data_utils import mtminibatches, pad_sequences, get_chunks
from .base_model import BaseModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

class MTLModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(MTLModel, self).__init__(config)
        self.idx_to_cbtag = {idx: tag for tag, idx in
                           self.config.vocab_cbtags.items()}
        self.idx_to_prptag = {idx: tag for tag, idx in
                           self.config.vocab_prptags.items()}
        self.idx_to_gentag = {idx: tag for tag, idx in
                           self.config.vocab_gentags.items()}

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")
        # shape=[batch size]
        # self.sentence_ids = tf.placeholder(tf.int32, shape=[None],
        #                 name="sents_ids")
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.cblabels = tf.placeholder(tf.int32, shape=[None, None],
                        name="cblabels")
        # MTL cb and event or prp label sshape = (batch size, max length of sentence in batch)
        self.prp_labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="prplabels")
        # MTL genre label shape = (batch size)
        self.gen_labels = tf.placeholder(tf.int32, shape=[None],
                        name="gen_labels")
        # self.gnr_label=tf.placeholder(tf.int32, shape=[None],
        #                 name="genlabels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, cblabels=None,prplabels=None,genlabels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if cblabels is not None:
            cblabels, _ = pad_sequences(cblabels, 0)
            feed[self.cblabels] = cblabels
        if prplabels is not None:
            prplabels, _ = pad_sequences(prplabels, 0)
            feed[self.prp_labels] = prplabels
        # if genlabels is not None:
        #     genlabels, _ = pad_sequences(genlabels, 0)
        if genlabels is not None:
            feed[self.gen_labels] = genlabels
        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                if self.config.embed_mode=="prp":
                    _word_embeddings = tf.Variable(
                            self.config.prp_embeddings,
                            name="_word_embeddings",
                            dtype=tf.float32,
                            trainable=self.config.train_embeddings)
                    nchars=self.config.nprpchars
                elif self.config.embed_mode=="ev":
                    _word_embeddings = tf.Variable(
                            self.config.ev_embeddings,
                            name="_word_embeddings",
                            dtype=tf.float32,
                            trainable=self.config.train_embeddings)
                    nchars=self.config.nevchars
                elif self.config.embed_mode=="mix":
                    _word_embeddings = tf.Variable(
                            self.config.mix_embeddings,
                            name="_word_embeddings",
                            dtype=tf.float32,
                            trainable=self.config.train_embeddings)
                    nchars=self.config.nmixchars

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")
        self.config.nchars=nchars
        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
    def add_hard_logits_op(self):
        """Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstmt"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("bi-lstmt1"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cellprp_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cellprp_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (outputprp_fw, outputprp_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cellprp_fw, cellprp_bw, output,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            outputprp = tf.concat([outputprp_fw, outputprp_bw], axis=-1)
            outputprp = tf.nn.dropout(outputprp, self.dropout)
        with tf.variable_scope("bi-lstmt2"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cellcb_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cellcb_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (outputlcb_fw, outputlcb_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cellcb_fw, cellcb_bw, output,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)

            outputlcb = tf.concat([outputlcb_fw, outputlcb_bw], axis=-1)
            outputlcb = tf.nn.dropout(outputlcb, self.dropout)
        with tf.variable_scope("bi-lstmt3"):
            #####this is for genre
            cellg_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,
                        state_is_tuple=True)
            cellg_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,
                        state_is_tuple=True)
            _outputg = tf.nn.bidirectional_dynamic_rnn(
                    cellg_fw, cellg_bw, output,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            _, ((_, outputg_fw), (_, outputg_bw)) = _outputg
            outputg = tf.concat([outputg_fw, outputg_bw], axis=-1)
            outputg = tf.nn.dropout(outputg, self.dropout)
        with tf.variable_scope("projt"):
            ###cb
            W2 = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ncbtags])
            b2 = tf.get_variable("b", shape=[self.config.ncbtags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(outputlcb)[1]
            outputcb = tf.reshape(outputlcb, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(outputcb, W2) + b2
            class_weight = tf.constant(self.config.weights_cblable)
            logits = tf.reshape(pred, [-1, nsteps, self.config.ncbtags])
            self.cblogits = tf.multiply(logits, class_weight)
        # self.cblogits = tf.reshape(pred, [-1, nsteps, self.config.ncbtags])

        ###prp
            Wp2 = tf.get_variable("W1", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.nprptags])
            bp2 = tf.get_variable("b1", shape=[self.config.nprptags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            npsteps = tf.shape(outputprp)[1]
            outputp = tf.reshape(outputprp, [-1, 2*self.config.hidden_size_lstm])
            predp = tf.matmul(outputp, Wp2) + bp2
            prplogits = tf.reshape(predp, [-1, npsteps, self.config.nprptags])
            class_prev_weight = tf.constant(self.config.weights_prevlable)
            self.prplogits = tf.multiply(prplogits, class_prev_weight)
        # self.prplogits = tf.reshape(predp, [-1, npsteps, self.config.nprptags])

        ###gen
            Wg2 = tf.get_variable("W2", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ngentags])
            bg2 = tf.get_variable("b2", shape=[self.config.ngentags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            ngsteps = tf.shape(outputg)[1]
            outputgen = tf.reshape(outputg, [-1, 2*self.config.hidden_size_lstm])
            predgen = tf.matmul(outputgen, Wg2) + bg2
            genlogits = tf.reshape(predgen, [-1, self.config.ngentags])
            class_gen_weight = tf.constant(self.config.weightgrn_lb)
            self.genlogits = tf.multiply(genlogits, class_gen_weight)
            # self.genlogits = tf.reshape(predgen, [-1, self.config.ngentags])

    def add_lcb_p_logits_op(self):
        """Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstmt"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("bi-lstmt1"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cellprp_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cellprp_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (outputprp_fw, outputprp_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cellprp_fw, cellprp_bw, output,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            outputprp = tf.concat([outputprp_fw, outputprp_bw], axis=-1)
            outputprp = tf.nn.dropout(outputprp, self.dropout)
        with tf.variable_scope("bi-lstmt2"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cellcb_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cellcb_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (outputlcb_fw, outputlcb_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cellcb_fw, cellcb_bw, output,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)

            outputlcb = tf.concat([outputlcb_fw, outputlcb_bw,outputprp], axis=-1)
            outputlcb = tf.nn.dropout(outputlcb, self.dropout)
        with tf.variable_scope("bi-lstmt3"):
            #####this is for genre
            cellg_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,
                        state_is_tuple=True)
            cellg_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,
                        state_is_tuple=True)
            _outputg = tf.nn.bidirectional_dynamic_rnn(
                    cellg_fw, cellg_bw, output,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            _, ((_, outputg_fw), (_, outputg_bw)) = _outputg
            outputg = tf.concat([outputg_fw, outputg_bw], axis=-1)
            outputg = tf.nn.dropout(outputg, self.dropout)
        with tf.variable_scope("projt"):
            ###cb
            W2 = tf.get_variable("W", dtype=tf.float32,
                    shape=[4*self.config.hidden_size_lstm, self.config.ncbtags])
            b2 = tf.get_variable("b", shape=[self.config.ncbtags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(outputlcb)[1]
            outputcb = tf.reshape(outputlcb, [-1, 4*self.config.hidden_size_lstm])
            pred = tf.matmul(outputcb, W2) + b2
            class_weight = tf.constant(self.config.weights_cblable)
            logits = tf.reshape(pred, [-1, nsteps, self.config.ncbtags])
            self.cblogits = tf.multiply(logits, class_weight)
        # self.cblogits = tf.reshape(pred, [-1, nsteps, self.config.ncbtags])

        ###prp
            Wp2 = tf.get_variable("W1", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.nprptags])
            bp2 = tf.get_variable("b1", shape=[self.config.nprptags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            npsteps = tf.shape(outputprp)[1]
            outputp = tf.reshape(outputprp, [-1, 2*self.config.hidden_size_lstm])
            predp = tf.matmul(outputp, Wp2) + bp2
            prplogits = tf.reshape(predp, [-1, npsteps, self.config.nprptags])
            class_prev_weight = tf.constant(self.config.weights_prevlable)
            self.prplogits = tf.multiply(prplogits, class_prev_weight)
        # self.prplogits = tf.reshape(predp, [-1, npsteps, self.config.nprptags])

        ###gen
            Wg2 = tf.get_variable("W2", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ngentags])
            bg2 = tf.get_variable("b2", shape=[self.config.ngentags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            ngsteps = tf.shape(outputg)[1]
            outputgen = tf.reshape(outputg, [-1, 2*self.config.hidden_size_lstm])
            predgen = tf.matmul(outputgen, Wg2) + bg2
            genlogits = tf.reshape(predgen, [-1, self.config.ngentags])
            class_gen_weight = tf.constant(self.config.weightgrn_lb)
            self.genlogits = tf.multiply(genlogits, class_gen_weight)
            # self.genlogits = tf.reshape(predgen, [-1, self.config.ngentags])

    def add_t2logits_op(self):
        """Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstmt"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("bi-lstmt2"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, output,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)

            outpput = tf.concat([output_fw, output_bw], axis=-1)
            outpput = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("bi-lstmt3"):
            #####this is for genre
            cellg_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,
                        state_is_tuple=True)
            cellg_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,
                        state_is_tuple=True)
            _outputg = tf.nn.bidirectional_dynamic_rnn(
                    cellg_fw, cellg_bw, output,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            _, ((_, outputg_fw), (_, outputg_bw)) = _outputg
            outputg = tf.concat([outputg_fw, outputg_bw], axis=-1)
            outputg = tf.nn.dropout(outputg, self.dropout)

        with tf.variable_scope("projt"):
            ###cb
            if self.config.usNN==False:
                if self.config.use_chars==False:
                    #we do this hence in reshaping the tensor it raises the error of mismatch size
                    self.config.hidden_size_lstm = self.config.dim_word

                W2 = tf.get_variable("W", dtype=tf.float32,
                                     shape=[self.config.hidden_size_lstm, self.config.ncbtags])
                b2 = tf.get_variable("b", shape=[self.config.ncbtags],
                                     dtype=tf.float32, initializer=tf.zeros_initializer())

                nsteps = tf.shape(self.word_embeddings)[1]
                outputcb = tf.reshape(self.word_embeddings, [-1, self.config.hidden_size_lstm])
                pred = tf.matmul(outputcb, W2) + b2
                class_weight = tf.constant(self.config.weights_cblable)
                logits = tf.reshape(pred, [-1, nsteps, self.config.ncbtags])
                self.cblogits = tf.multiply(logits, class_weight)
                ###prp

                Wp2 = tf.get_variable("W1", dtype=tf.float32,
                                      shape=[2 * self.config.hidden_size_lstm, self.config.nprptags])
                bp2 = tf.get_variable("b1", shape=[self.config.nprptags],
                                      dtype=tf.float32, initializer=tf.zeros_initializer())
                npsteps = tf.shape(output)[1]
                outputp = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
                predp = tf.matmul(outputp, Wp2) + bp2
                prplogits = tf.reshape(predp, [-1, npsteps, self.config.nprptags])
                class_prev_weight = tf.constant(self.config.weights_prevlable)
                self.prplogits = tf.multiply(prplogits, class_prev_weight)
                # self.prplogits = tf.reshape(predp, [-1, npsteps, self.config.nprptags])

                ###gen
                Wg2 = tf.get_variable("W2", dtype=tf.float32,
                                      shape=[2 * self.config.hidden_size_lstm, self.config.ngentags])
                bg2 = tf.get_variable("b2", shape=[self.config.ngentags],
                                      dtype=tf.float32, initializer=tf.zeros_initializer())
                ngsteps = tf.shape(outputg)[1]
                outputgen = tf.reshape(outputg, [-1, 2 * self.config.hidden_size_lstm])
                predgen = tf.matmul(outputgen, Wg2) + bg2
                genlogits = tf.reshape(predgen, [-1, self.config.ngentags])
                class_gen_weight = tf.constant(self.config.weightgrn_lb)
                self.genlogits = tf.multiply(genlogits, class_gen_weight)


            else:
                W2 = tf.get_variable("W", dtype=tf.float32,
                        shape=[2*self.config.hidden_size_lstm, self.config.ncbtags])
                b2 = tf.get_variable("b", shape=[self.config.ncbtags],
                        dtype=tf.float32, initializer=tf.zeros_initializer())


                nsteps = tf.shape(output)[1]
                outputcb = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
                pred = tf.matmul(outputcb, W2) + b2
                class_weight = tf.constant(self.config.weights_cblable)
                logits = tf.reshape(pred, [-1, nsteps, self.config.ncbtags])
                self.cblogits = tf.multiply(logits, class_weight)
            # self.cblogits = tf.reshape(pred, [-1, nsteps, self.config.ncbtags])

            ###prp
                Wp2 = tf.get_variable("W1", dtype=tf.float32,
                        shape=[2*self.config.hidden_size_lstm, self.config.nprptags])
                bp2 = tf.get_variable("b1", shape=[self.config.nprptags],
                        dtype=tf.float32, initializer=tf.zeros_initializer())
                npsteps = tf.shape(output)[1]
                outputp = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
                predp = tf.matmul(outputp, Wp2) + bp2
                prplogits = tf.reshape(predp, [-1, npsteps, self.config.nprptags])
                class_prev_weight = tf.constant(self.config.weights_prevlable)
                self.prplogits = tf.multiply(prplogits, class_prev_weight)
            # self.prplogits = tf.reshape(predp, [-1, npsteps, self.config.nprptags])

            ###gen
                Wg2 = tf.get_variable("W2", dtype=tf.float32,
                        shape=[2*self.config.hidden_size_lstm, self.config.ngentags])
                bg2 = tf.get_variable("b2", shape=[self.config.ngentags],
                        dtype=tf.float32, initializer=tf.zeros_initializer())
                ngsteps = tf.shape(outputg)[1]
                outputgen = tf.reshape(outputg, [-1, 2*self.config.hidden_size_lstm])
                predgen = tf.matmul(outputgen, Wg2) + bg2
                genlogits = tf.reshape(predgen, [-1, self.config.ngentags])
                class_gen_weight = tf.constant(self.config.weightgrn_lb)
                self.genlogits = tf.multiply(genlogits, class_gen_weight)
                # self.genlogits = tf.reshape(predgen, [-1, self.config.ngentags])


    def add_t3logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstmt"):
            # self.t2_input=tf.concat([self.word_embeddings,self.logits])
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("projt"):
            W2 = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ncbtags])
            b2 = tf.get_variable("b", shape=[self.config.ncbtags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W2) + b2
            self.mtlogits = tf.reshape(pred, [-1, nsteps, self.config.ncbtags])

    def add_t1loss_op(self):
        """Defines the loss"""
        task_wigths=self.config.weights_task
        if self.config.use_crf:
            with tf.variable_scope('1stloss'):
                t1log_likelihood, t1trans_params = tf.contrib.crf.crf_log_likelihood(
                        self.cblogits, self.cblabels, self.sequence_lengths)
                self.t1trans_params = t1trans_params # need to evaluate it for decoding
                self.t1loss = tf.reduce_mean(-t1log_likelihood)
            with tf.variable_scope('2sndloss'):
                t2log_likelihood, t2trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.prplogits, self.prp_labels, self.sequence_lengths)
                self.t2trans_params = t2trans_params  # need to evaluate it for decoding
                self.t2loss = tf.reduce_mean(-t2log_likelihood)
            with tf.variable_scope('3sndloss'):
                t3log_likelihood = tf.losses.sparse_softmax_cross_entropy(logits=self.genlogits,labels=self.gen_labels)
                self.t3loss = tf.reduce_mean(t3log_likelihood)

                # need to evaluate it for decoding


        #added for task weigths
            if len(self.config.model_mode)==3:
                self.mtloss=task_wigths[0]*self.t1loss+task_wigths[1]*self.t2loss+task_wigths[2]*self.t3loss
            elif len(self.config.model_mode)==2:
                self.mtloss=task_wigths[0]*self.t1loss+task_wigths[1]*self.t2loss
            elif len(self.config.model_mode) == 1:
                self.mtloss = task_wigths[0] * self.t1loss
            elif len(self.config.model_mode) == 4:
                self.mtloss = task_wigths[2]*self.t3loss
            elif len(self.config.model_mode) == 5:
                self.mtloss = task_wigths[1]*self.t2loss
            elif len(self.config.model_mode) == 6:
                self.mtloss = task_wigths[0]*self.t1loss+task_wigths[2]*self.t3loss
            elif len(self.config.model_mode) == 7:
                self.mtloss = task_wigths[1]*self.t2loss+task_wigths[2]*self.t3loss
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.cblogits, labels=self.cblabels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.mtloss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("mtloss", self.mtloss)

        # now self.sess is defined and vars are init

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.cblogits, axis=-1),
                    tf.int32)

    def trfbuild(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_t2logits_op()
        self.add_pred_op()
        self.add_t1loss_op()
        # self.add_t2loss_op()


        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.mtloss,
                self.config.clip)
        self.initialize_session()
    def hardmtlbuild(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_hard_logits_op()
        self.add_pred_op()
        self.add_t1loss_op()
        # self.add_t2loss_op()


        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.mtloss,
                self.config.clip)
        self.initialize_session()
    def LCBPmtlbuild(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_lcb_p_logits_op()
        self.add_pred_op()
        self.add_t1loss_op()
        # self.add_t2loss_op()


        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.mtloss,
                self.config.clip)
        self.initialize_session()
    def evbuild(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_t2logits_op()
        self.add_pred_op()
        self.add_t1loss_op()
        # self.add_t2loss_op()


        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.mtloss,
                self.config.clip)
        self.initialize_session()
    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            t1viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.cblogits, self.t1trans_params], feed_dict=fd)
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            t1logits, t1trans_params = self.sess.run(
                    [self.prplogits, self.t2trans_params], feed_dict=fd)
            for t1logit, t1sequence_length in zip(t1logits, sequence_lengths):
                t1logit = t1logit[:t1sequence_length]  # keep only the valid steps
                t1viterbi_seq, t1viterbi_score = tf.contrib.crf.viterbi_decode(
                    t1logit, t1sequence_length)
                t1viterbi_sequences += [t1viterbi_seq]
        # iterate over the sentences because no batching in vitervi_decode

            labels_gen_pred = self.sess.run(self.genlogits, feed_dict=fd)

            return viterbi_sequences, t1viterbi_sequences, labels_gen_pred, sequence_lengths

        else:
            viterbi_sequences = self.sess.run(self.cblogits, feed_dict=fd)
            t1viterbi_sequences = self.sess.run(self.prplogits, feed_dict=fd)
            labels_gen_pred = self.sess.run(self.genlogits, feed_dict=fd)

            return viterbi_sequences, t1viterbi_sequences, labels_gen_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        # iterate over dataset
        for i, (words, cblabels,prplabels,genlabels,wordr) in enumerate(mtminibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, cblabels, prplabels,genlabels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.mtloss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = "model_num: "+str(self.config.model_name)+"  "+" - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
        self.logger.info("learning Rate"+"_"+str(self.config.lr)+"_"+str(self.config.dropout)+" With CRF or Without:"+str(self.config.use_crf))

        return metrics["f1"]
    def test_models(self,test):
        """Testing the pretrained model on the test set

        Args:
            test: dataset

        """
        #name of training

        metrics = self.run_evaluate(test)
        msg = "model_num: "+str(self.config.model_name)+"  "+" - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
        self.logger.info("learning Rate"+"_"+str(self.config.lr)+"_"+str(self.config.dropout)+" With CRF or Without:"+str(self.config.use_crf))

        return metrics["f1"]

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        cb_tagList, evprp_list, genr_lbl, siz = self.predict_batch([words])

        # pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_cbtag[idx] for idx in list(cb_tagList[0])]
        prppreds=[self.idx_to_prptag[idx] for idx in list(evprp_list[0])]
        genpreds=[self.idx_to_gentag[np.argmax(genr_lbl[0])]]

        return [preds,prppreds,genpreds]
