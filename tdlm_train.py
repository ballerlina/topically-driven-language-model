"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Modified By:    Carolina Zheng
Date:           Jul 16
"""

import argparse
import sys
import codecs
import random
import time
import os
import pdb
import cPickle
import tensorflow as tf
import numpy as np
import gensim.models as g
import tdlm_config as cf
from ltlm_data import SOS, EOS, PAD, process_dataset, get_batch_v2
from ltlm_util import Config
from util import init_embedding
from tdlm_model import TopicModel as TM
from tdlm_model import LanguageModel as LM

#parser arguments
desc = "trains neural topic language model on a document collection (experiment settings defined in cf.py)"
parser = argparse.ArgumentParser(description=desc)
args = parser.parse_args()


###########
#functions#
###########
def run_epoch(data, models, is_training, ltlm_cf):
    ####unsupervised topic and language model training####
    docs_segmented, doc_bows, doc_num_tokens = train_data
    batch_idx = 0
    prev_doc_idxs = None
    prev_sequence_idxs = None
    prev_running_bows = np.zeros((cf.batch_size, num_tm_words))

    #set training and cost ops for topic and language model training
    tm_cost_ops = (tf.no_op(), tf.no_op(), tf.no_op(), tf.no_op())
    lm_cost_ops = (tf.no_op(), tf.no_op(), tf.no_op(), tf.no_op())
    if models[0] != None:
        tm_cost_ops = (models[0].tm_cost, (models[0].tm_train_op if is_training else tf.no_op()), tf.no_op(), tf.no_op())
    if models[1] != None:
        lm_cost_ops = (tf.no_op(), tf.no_op(), models[1].lm_cost, (models[1].lm_train_op if is_training else tf.no_op()))

    start_time = time.time()
    lm_costs, tm_costs, lm_words, tm_words = 0.0, 0.0, 0.0, 0.0
    while True:
        (
            data,
            bows,
            target,
            mask,
            prev_doc_idxs,
            prev_sequence_idxs,
            is_last_sequence,
        ) = get_batch_v2(
            ltlm_cf,
            cf.batch_size,
            docs_segmented,
            doc_bows,
            prev_doc_idxs,
            prev_sequence_idxs,
            prev_running_bows,
        )
        if data is None:
            # if epoch == 1:
            #     print(f"{batch_idx} total batches")
            break
        batch_idx += 1

        # TODO: Initialize LM hidden states if saving

        # Train LM
        # data = (bsz x seq_len)
        model = models[1]
        feed_dict = {model.x: data, model.y: target, model.lm_mask: mask}
        if cf.topic_number > 0:
            feed_dict.update({model.doc: bows, model.tag: None})

        tm_cost, _, lm_cost, _ = sess.run(lm_cost_ops, feed_dict)

        # Train TM
        if models[0] is not None:
            model = models[0]
            # TODO: target should be different for TM
            feed_dict = {model.y: target, model.tm_mask: mask, model.doc: bows, model.tag: None}
            tm_cost, _, lm_cost, _ = sess.run(tm_cost_ops, feed_dict)

        if tm_cost != None:
            tm_costs += tm_cost * cf.batch_size #keep track of full batch loss (not per example batch loss)
            tm_words += np.sum(mask)

        if lm_cost != None:
            lm_costs += lm_cost * cf.batch_size
            lm_words += np.sum(mask)

        #print progress
        output_string = "%d: tm ppl = %.3f; lm ppl = %.3f; word/sec = %.1f" % \
            (batch_idx, np.exp(tm_costs/max(tm_words, 1.0)), np.exp(lm_costs/max(lm_words, 1.0)),  \
            float(tm_words + lm_words)/(time.time()-start_time))
        print_progress(batch_idx, is_training, output_string)

    if cf.verbose:
        sys.stdout.write("\n")
            
    return np.exp(lm_costs/max(lm_words, 1.0))

def print_progress(bi, is_training, output_string):
    if ((bi % 200) == 0) and cf.verbose:
        if is_training:
            sys.stdout.write("TRAIN ")
        else:
            sys.stdout.write("VALID ")
        sys.stdout.write(output_string + "\r")
        sys.stdout.flush()

######
#main#
######
#set the seeds
random.seed(cf.seed)
np.random.seed(cf.seed)
#utf-8 output
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

#set topic vector size and load word embedding model if given
if cf.word_embedding_model:
    print "Loading word embedding model..."
    mword = g.KeyedVectors.load_word2vec_format(cf.word_embedding_model, binary=True)
    cf.word_embedding_size = mword.vector_size

# #labels given for documents
train_labels, valid_labels, num_classes = None, None, 0
# #tags given for documents
train_tags, valid_tags, tagxid, tag_len = None, None, {} , 0

# Load data
if cf.stopwords is not None:
    with open(cf.stopwords, "r") as f:
        stopwords = set(f.read().splitlines())
else:
    stopwords = {}

train_data, val_data, test_data, vocab, num_tm_words = process_dataset(
        stopwords, cf.data_path, cf.lm_sent_len, cf.doc_len, reproduce_tdlm=True
)

# print some statistics of the data
print "Vocab size =", len(vocab)

config_attrs = {
    "model_type": "TDLM",
    "num_tm_words": num_tm_words,
    "max_seqlen": cf.lm_sent_len,
    "pad_idx": vocab[PAD],
    "use_all_bows": False,
    "eval_false": True,
    "reset_hidden": True,
}
ltlm_cf = Config(**config_attrs)

#train model
with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(cf.seed)
    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope("model", reuse=None, initializer=initializer):
        tm_train = TM(is_training=True, vocab_size=len(vocab), batch_size=cf.batch_size, \
            num_steps=cf.tm_sent_len, num_classes=num_classes, config=cf) if cf.topic_number > 0 else None
        lm_train = LM(is_training=True, vocab_size=len(vocab), batch_size=cf.batch_size, \
            num_steps=cf.lm_sent_len, config=cf, reuse_conv_variables=True) \
            if cf.rnn_hidden_size > 0  else None

    with tf.variable_scope("model", reuse=True, initializer=initializer):
        tm_valid = TM(is_training=False, vocab_size=len(vocab), batch_size=cf.batch_size, \
            num_steps=cf.tm_sent_len, num_classes=num_classes, config=cf) if cf.topic_number > 0 else None
        lm_valid = LM(is_training=False, vocab_size=len(vocab), batch_size=cf.batch_size, \
            num_steps=cf.lm_sent_len, config=cf) if cf.rnn_hidden_size > 0 else None

    tf.global_variables_initializer().run()

    #initialise word embedding
    if cf.word_embedding_model:
        word_emb = init_embedding(mword, vocab.idxvocab)
        if cf.rnn_hidden_size > 0:
            sess.run(lm_train.lstm_word_embedding.assign(word_emb))
        if cf.topic_number > 0:
            sess.run(tm_train.conv_word_embedding.assign(word_emb))

    #save model every epoch
    if cf.save_model:
        if not os.path.exists(os.path.join(cf.output_dir, cf.output_prefix)):
            os.makedirs(os.path.join(cf.output_dir, cf.output_prefix))
        #create saver object to save model
        saver = tf.train.Saver()

    #train model
    prev_ppl = None
    for i in xrange(cf.epoch_size):
        print "\nEpoch =", i
        #run a train epoch
        run_epoch(train_data, (tm_train, lm_train), True, ltlm_cf)
        #run a valid epoch
        curr_ppl = run_epoch(val_data, (tm_valid, lm_valid), False, ltlm_cf)
    
        if cf.save_model:
            if (i < 5) or (prev_ppl == None) or (curr_ppl < prev_ppl):
                saver.save(sess, os.path.join(cf.output_dir, cf.output_prefix, "model.ckpt"))
                prev_ppl = curr_ppl
            else:
                saver.restore(sess, os.path.join(cf.output_dir, cf.output_prefix, "model.ckpt"))
                print "\tNew valid performance > prev valid performance: restoring previous parameters..."

    #print top-N words from topics
    if cf.topic_number > 0:
        print "\nTopics\n======"
        topics, entropy = tm_train.get_topics(sess, topn=20)
        for ti, t in enumerate(topics):
            print "Topic", ti, "[", ("%.2f" % entropy[ti]), "] :", " ".join([ vocab[item] for item in t ])

    #generate some random sentences
    if cf.rnn_hidden_size > 0:
        print "\nRandom Generated Sentences\n=========================="
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mgen = LM(is_training=False, vocab_size=len(vocab), batch_size=1, num_steps=1, config=cf, \
                reuse_conv_variables=True)
        for temp in [1.0, 0.75, 0.5]:
            print "\nTemperature =", temp
            for _ in xrange(10):
                #select a random topic
                if cf.topic_number > 0:
                    topic = random.randint(0, cf.topic_number-1)
                    print "\tTopic", topic, ":",
                else:
                    topic = -1
                    print "\t",

                s = mgen.generate_on_topic(sess, topic, vocab[SOS], temp, cf.lm_sent_len+10, \
                    vocab[EOS])
                s = [ vocab[item] for item in s ]
                print " ".join(s)

    #save model vocab and configurations
    if cf.save_model:
        #vocabulary information
        cPickle.dump((vocab, num_tm_words), \
            open(os.path.join(cf.output_dir, cf.output_prefix, "vocab.pickle"), "w"))

        #create a dictionary object for config
        cf_dict = {}
        for k,v in vars(cf).items():
            if not k.startswith("__"):
                cf_dict[k] = v
        cPickle.dump(cf_dict, open(os.path.join(cf.output_dir, cf.output_prefix, "config.pickle"), "w"))
