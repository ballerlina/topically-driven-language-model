"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Modified by:    Carolina Zheng
Date:           Oct 16
"""

import argparse
import sys
import os
import cPickle
import math
import codecs
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tdlm_model import TopicModel as TM
from tdlm_model import LanguageModel as LM
from util import get_batch_doc
from gensim import matutils
from ltlm_data import process_dataset, get_batch_v2, PAD, SOS, EOS
from ltlm_util import Config

#parser arguments
desc = "Given a trained TDLM model, perform various test inferences"
parser = argparse.ArgumentParser(description=desc)

###################
#optional argument#
###################
parser.add_argument("-m", "--model_dir", required=True, help="directory of the saved model")
parser.add_argument("-d", "--input_doc", help="input file containing the test documents")
parser.add_argument("-l", "--input_label", help="input file containing the test labels")
parser.add_argument("-t", "--input_tag", help="input file containing the test tags")
parser.add_argument("--print_perplexity", help="print topic and language model perplexity of the input test documents", \
    action="store_true")
parser.add_argument("--print_acc", help="print supervised classification accuracy", action="store_true")
parser.add_argument("--output_topic", help="output file to save the topics (prints top-N words of each topic)")
parser.add_argument("--output_topic_dist", \
    help="output file to save the topic distribution of input docs (npy format)")
parser.add_argument("--output_tag_embedding", \
    help="output tag embeddings to file (npy format)")
parser.add_argument("--gen_sent_on_topic", help="generate sentences conditioned on topics")
parser.add_argument("--gen_sent_on_doc", help="generate sentences conditioned on input test documents")

args = parser.parse_args()

#parameters
topn=10 #number of top-N words to print for each topic
gen_temps = [1.0, 0.75] #temperatures for generation
gen_num = 3 #number of generated sentences
debug = False

###########
#functions#
###########

def compute_dt_dist(docs, labels, tags, model, max_len, batch_size, pad_id, vocab, output_file):
    #generate batches
    num_batches = int(math.ceil(float(len(docs)) / batch_size))
    dt_dist = []
    t = []
    combined = []
    docid = 0
    for i in xrange(num_batches):
        x, _, _, t, s = get_batch_doc(docs, labels, tags, i, max_len, cf.tag_len, batch_size, pad_id)
        attention, mean_topic = sess.run([model.attention, model.mean_topic], {model.doc: x, model.tag: t})
        dt_dist.extend(attention[:s])

        if debug:
            for si in xrange(s):
                d = x[si]
                print "\n\nDoc", docid, "=", " ".join([vocab.get_word(item) for item in d if (item != pad_id)])
                sorted_dist = matutils.argsort(attention[si], reverse=True)
                for ti in sorted_dist:
                    print "Topic", ti, "=", attention[si][ti]
                docid += 1

    np.save(open(output_file, "w"), dt_dist)

def run_epoch(data, (tm, lm), pad_id, cf, ltlm_cf):
    docs_segmented, doc_bows, doc_num_tokens = data
    batch_idx = 0
    prev_doc_idxs = None
    prev_sequence_idxs = None
    prev_running_bows = np.zeros((cf.batch_size, num_tm_words))
    lm_costs, lm_words = 0.0, 0.0
    if tm != None:
        tm_costs, tm_words = 0.0, 0.0

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
            break
        batch_idx += 1

        feed_dict = {lm.x: data, lm.y: target, lm.lm_mask: mask}
        if cf.topic_number > 0:
            feed_dict.update({lm.doc: bows, lm.tag: None})

        lm_cost = sess.run(lm.lm_cost, feed_dict)
        lm_costs += lm_cost * cf.batch_size
        lm_words += np.sum(mask)

        if tm != None:
            # TODO: Compute TM ppl
            pass

    print("Num batches: {}".format(batch_idx))
    print "test language model perplexity = %.3f" % (np.exp(lm_costs/lm_words))

    if tm != None:
        print "\ntest topic model perplexity = %.3f" % (np.exp(tm_costs/tm_words))


def run_epoch_doc(docs, labels, tm, pad_id, cf):
    batches = int(math.ceil(float(len(docs))/cf.batch_size))
    accs = []
    for b in xrange(batches):
        d, y, m, t, num_docs = get_batch_doc(docs, labels, tags, b, cf.doc_len, cf.tag_len, cf.batch_size, pad_id)
        prob = sess.run(tm.sup_probs, {tm.doc:d, tm.label:y, tm.sup_mask: m, tm.tag: t})
        pred = np.argmax(prob, axis=1)
        accs.extend(pred[:num_docs] == y[:num_docs])

    print "\ntest classification accuracy = %.3f" % np.mean(accs)

def gen_sent_on_topic(vocab, start_symbol, end_symbol, cf):
    output = codecs.open(args.gen_sent_on_topic, "w", "utf-8")
    topics, entropy = tm.get_topics(sess, topn=topn)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mgen = LM(is_training=False, vocab_size=len(vocab), batch_size=1, num_steps=1, config=cf, \
            reuse_conv_variables=True)

    for t in range(cf.topic_number):
        output.write("\n" + "="*100 + "\n")
        output.write("Topic " +  str(t) + ":\n")
        output.write(" ".join([ vocab.get_word(item) for item in topics[t] ]) + "\n\n")

        output.write("\nSentence generation (greedy; argmax):" + "\n")
        s = mgen.generate_on_topic(sess, t, vocab[start_symbol], 0, cf.lm_sent_len+10, vocabxid[end_symbol])
        output.write("[0] " + " ".join([ vocab.get_word(item) for item in s ]) + "\n")
        
        for temp in gen_temps:
            output.write("\nSentence generation (random; temperature = " + str(temp) + "):\n")
            for i in xrange(gen_num):
                s = mgen.generate_on_topic(sess, t, vocab[start_symbol], temp, cf.lm_sent_len+10, \
                    vocab[end_symbol])
                output.write("[" + str(i) + "] " +  " ".join([ vocab.get_word(item) for item in s ]) + "\n")

def gen_sent_on_doc(docs, tags, vocab, start_symbol, end_symbol, cf):
    topics, _ = tm.get_topics(sess, topn=topn)
    topics = [ " ".join([vocab.get_word(w) for w in t]) for t in topics ]
    doc_text = [ item.replace("\t", "\n") for item in codecs.open(args.input_doc, "r", "utf-8").readlines() ]
    output = codecs.open(args.gen_sent_on_doc, "w", "utf-8")
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mgen = LM(is_training=False, vocab_size=len(vocab), batch_size=1, num_steps=1, config=cf, \
            reuse_conv_variables=True)

    for d in range(len(docs)):
        output.write("\n" + "="*100 + "\n")
        output.write("Doc " +  str(d) +":\n")
        output.write(doc_text[d])

        doc, _, _, t, _ = get_batch_doc(docs, None, tags, d, cf.doc_len, cf.tag_len, 1, vocabxid[pad_symbol])
        best_topics, best_words = mgen.get_topics_on_doc(sess, doc, t, topn)
        
        output.write("\nRepresentative topics:\n")
        output.write("\n".join([ ("[%.3f] %s: %s" % (item[1],str(item[0]).zfill(3),topics[item[0]])) \
            for item in best_topics ]) + "\n")

        output.write("\nRepresentative words:\n")
        output.write("\n".join([ ("[%.3f] %s" % (item[1], vocab.get_word(item[0]))) for item in best_words ]) + "\n")

        output.write("\nSentence generation (greedy; argmax):" + "\n")
        s = mgen.generate_on_doc(sess, doc, t, vocab[start_symbol], 0, cf.lm_sent_len+10, vocab[end_symbol])
        output.write("[0] " + " ".join([ vocab.get_word(item) for item in s ]) + "\n")

        for temp in gen_temps:
            output.write("\nSentence generation (random; temperature = " + str(temp) + "):\n")

            for i in xrange(gen_num):
                s = mgen.generate_on_doc(sess, doc, t, vocab[start_symbol], temp, cf.lm_sent_len+10, \
                    vocab[end_symbol])
                output.write("[" + str(i) + "] " + " ".join([ vocab.get_word(item) for item in s ]) + "\n")
######
#main#
######

#load the vocabulary
vocab, num_tm_words = cPickle.load(open(os.path.join(args.model_dir, "vocab.pickle")))
pad_symbol, start_symbol, end_symbol = PAD, SOS, EOS

#load config
cf_dict = cPickle.load(open(os.path.join(args.model_dir, "config.pickle")))
if "num_classes" not in cf_dict:
    cf_dict["num_classes"] = 0
if "num_tags" not in cf_dict:
    cf_dict["num_tags"] = 0
    cf_dict["tag_len"] = 0
    cf_dict["tag_embedding_size"] = 0
ModelConfig = namedtuple("ModelConfig", " ".join(cf_dict.keys()))
cf = ModelConfig(**cf_dict)

#parse and collect the documents
# if args.input_doc:
#     sents, docs, docids, stats = gen_data(vocab, dummy_symbols, tm_ignore, args.input_doc, \
#         cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)
if cf.stopwords is not None:
    with open(cf.stopwords, "r") as f:
        stopwords = set(f.read().splitlines())
else:
    stopwords = {}

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

_, _, test_data, vocab, num_tm_words = process_dataset(
    stopwords,
    cf.data_path,
    cf.lm_sent_len,
    cf.doc_len,
    reproduce_tdlm=True,
    vocab=vocab,
    test_only=True,
)
print "Vocab size =", len(vocab)

labels, tags = None, None

with tf.Graph().as_default(), tf.Session() as sess:
    initializer = tf.contrib.layers.xavier_initializer(seed=cf.seed)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        tm = TM(is_training=False, vocab_size=len(vocab), batch_size=cf.batch_size, \
            num_steps=cf.tm_sent_len, num_classes=cf.num_classes, config=cf) if cf.topic_number > 0 else None
        lm = LM(is_training=False, vocab_size=len(vocab), batch_size=cf.batch_size, \
            num_steps=cf.lm_sent_len, config=cf, reuse_conv_variables=True) if cf.rnn_hidden_size > 0 else None

    #load tensorflow model
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(args.model_dir, "model.ckpt"))

    #compute topic distribution of input documents
    if args.output_topic_dist:
        if args.input_doc == None:
            sys.stderr.write("Error: --output_topic_dist option requires --input_doc\n")
            raise SystemExit
        # TODO: Fix
        compute_dt_dist(docs[0], labels, tags, tm, cf.doc_len, cf.batch_size, vocab[pad_symbol], vocab, \
            args.output_topic_dist)

    #print topics
    if args.output_topic:
        topics, entropy = tm.get_topics(sess, topn=topn)
        output = codecs.open(args.output_topic, "w", "utf-8")
        for ti, t in enumerate(topics):
            output.write(" ".join([ vocab.get_word(item) for item in t ]) + "\n")

    #compute test perplexities
    if args.print_perplexity:
        if args.input_doc == None:
            sys.stderr.write("Error: --print_perplexity requires --input_doc\n")
            raise SystemExit
        run_epoch(test_data, (tm, lm), vocab[pad_symbol], cf, ltlm_cf)

    #generate sentences conditioned on topics
    if args.gen_sent_on_topic:
        gen_sent_on_topic(vocab, start_symbol, end_symbol, cf)

    #generate sentences conditioned on documents
    if args.gen_sent_on_doc:
        if args.input_doc == None:
            sys.stderr.write("Error: --gen_sent_on_doc option requires --input_doc\n")
            raise SystemExit
        # TODO: Fix
        gen_sent_on_doc(docs[0], tags, vocab, start_symbol, end_symbol, cf)
