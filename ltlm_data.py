import os
import pdb
import re
import numpy as np
from collections import Counter, OrderedDict

SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"
MIN_FREQ = 10
MIN_DOC_FREQ = 100
MAX_FREQ_PCT = 0.001


class Vocab:
    def __init__(self, ordered_dict):
        def build():
            self.vocabxfreq = ordered_dict
            self.vocabxid = {}
            self.idxvocab = {}

            for i, (word, _) in enumerate(ordered_dict.items()):
                self.vocabxid[word] = i
                self.idxvocab[i] = word

        build()
        self.unk = None

    def __len__(self):
        return len(self.vocabxid)

    def __getitem__(self, key):
        if key in self.vocabxid:
            return self.vocabxid[key]
        else:
            return self.vocabxid[self.unk]

    def get_word(self, id):
        if id in self.idxvocab.keys():
            return self.idxvocab[id]
        else:
            return None

    def set_unk(self, unk):
        self.unk = unk


# tdlm data: each line is a document, already tokenized
# sentences are separated by \t and the end of the document has an extra \n
def get_tdlm_iter(data_path, split="train"):
    with open(os.path.join(data_path, "{}.txt".format(split)), "r") as f:
        docs = f.readlines()
        lines = [line for doc in docs for line in doc.split("\t")]

        if split == "train":
            counter = Counter()
            for doc in docs:
                counter.update(set(re.split(" |\t", doc.strip())))
            return docs, lines, counter
        else:
            return docs, lines


def create_vocab(counter, doc_counter, stopwords):
    filtered = {}

    for token, freq in counter.items():
        if freq >= MIN_FREQ:
            filtered[token] = freq

    filtered = Counter(filtered)

    stopwords = set([stopword for stopword in stopwords if stopword in filtered])
    freqwords = set(
        [item[0] for item in filtered.most_common(int(len(filtered) * MAX_FREQ_PCT))]
    )
    freqwords_doc = {
        item
        for item, doc_freq in doc_counter.items()
        if doc_freq < MIN_DOC_FREQ and item in filtered
    }
    alpha_check = re.compile("[a-zA-Z]")
    symbols = set(
        [
            token
            for token in filtered.keys()
            if (
                alpha_check.search(token) is None
                or token.startswith("'")
                or token.startswith("http://")
            )
        ]
    )
    dummy_symbols = [SOS, EOS, PAD, UNK]
    ignore = stopwords | freqwords | freqwords_doc | symbols | set(dummy_symbols)
    if "n't" in filtered:
        ignore.add("n't")

    def move_to_end(ordered_dict, key):
        value = ordered_dict[key]
        del ordered_dict[key]
        ordered_dict[key] = value

    filtered_ordered = OrderedDict(filtered)
    for token in ignore:
        if token in filtered_ordered:
            move_to_end(filtered_ordered, token)
        else:
            filtered_ordered[token] = 1

    num_tm_words = len(filtered_ordered) - len(ignore)
    items = list(filtered_ordered.items())
    sorted_tm = sorted(items[:num_tm_words], key=lambda x: (x[1], x[0]), reverse=True)
    sorted_ignore = sorted(
        items[num_tm_words:], key=lambda x: (x[1], x[0]), reverse=True
    )
    final_ordered = OrderedDict()

    for k, v in sorted_tm:
        final_ordered[k] = v
    for k, v in sorted_ignore:
        final_ordered[k] = v

    vocab = Vocab(final_ordered)
    vocab.set_unk(UNK)

    return vocab, len(vocab) - len(ignore)


# preprocessing follows TDLM (https://github.com/jhlau/topically-driven-language-model)
def process_dataset(
    stopwords,
    data_path,
    sequence_length,
    doc_length,
    reproduce_tdlm=False,
    vocab=None,
    val_only=False,
    test_only=False,
):
    train_docs, train_iter, doc_counter = get_tdlm_iter(data_path, split="train")
    val_docs, _ = get_tdlm_iter(data_path, split="valid")
    test_docs, _ = get_tdlm_iter(data_path, split="test")

    counter = Counter()

    for line in train_iter:
        counter.update(line.strip().split(" "))

    new_vocab, num_tm_words = create_vocab(counter, doc_counter, stopwords)

    if vocab is None:
        vocab = new_vocab

    def create_sequence_context_tuples_v2(docs):
        # doc_bows = np.zeros((len(docs), num_tm_words))
        doc_bows = np.full((len(docs), doc_length + sequence_length), vocab[PAD])
        doc_num_tokens = np.zeros(len(docs))
        docs_segmented = []
        sequence_count = 0

        for idx, doc in enumerate(docs):
            # if idx >= 100:
            #     break
            doc = SOS + " " + doc[:-1] + " " + EOS
            sent_delim_str = " " + EOS + " " if reproduce_tdlm else " "
            doc = doc.replace("\t", sent_delim_str).split(" ")
            doc_tokens = []
            bows_idx = 0
            for word in doc:
                token = vocab[word]
                if token < num_tm_words and bows_idx < len(doc_bows[idx]):
                    doc_bows[idx][bows_idx] += 1
                    bows_idx += 1
                doc_tokens.append(token)
            doc_num_tokens[idx] = len(doc_tokens)
            sequences = []
            docs_segmented.append(sequences)

            for start_idx in range(0, len(doc_tokens), sequence_length):
                sequence = np.full((sequence_length + 1,), vocab[PAD])
                end_idx = min(
                    len(doc_tokens), start_idx + sequence_length + 1
                )  # add extra token for target
                num_non_pad_tokens = end_idx - start_idx
                sequence[:num_non_pad_tokens] = doc_tokens[start_idx:end_idx]
                bows = np.zeros(num_tm_words)
                for token in sequence[:num_non_pad_tokens]:
                    if token < num_tm_words:
                        bows[token] += 1
                sequences.append((sequence, bows))
                sequence_count += 1

        print(
            "Finished loading data: {} sequences and {} docs.".format(sequence_count, len(docs_segmented))
        )

        return docs_segmented, doc_bows, doc_num_tokens

    train_data = (
        None if val_only or test_only else create_sequence_context_tuples_v2(train_docs)
    )
    val_data = None if test_only else create_sequence_context_tuples_v2(val_docs)
    test_data = None if val_only else create_sequence_context_tuples_v2(test_docs)

    return train_data, val_data, test_data, vocab, num_tm_words


# TODO: Implement returning bows correctly (bows should be tokens corresponding to first 300 words in document, minus current sequence)
def get_batch_v2(
    cf,
    batch_size,
    docs_segmented,
    doc_bows,
    prev_doc_idxs,
    prev_sequence_idxs,
    prev_running_bows,
    evaluate=False,
):
    sequences = []
    bows = []
    masks = []
    cur_doc_idxs = []
    cur_sequence_idxs = []
    is_last_sequence = np.zeros(batch_size)
    next_doc_idx = 0 if prev_doc_idxs is None else max(prev_doc_idxs) + 1

    for i in range(batch_size):
        if prev_doc_idxs is None:
            if len(docs_segmented) < batch_size:
                raise Exception("# documents smaller than batch size")
            doc_idx = next_doc_idx
            next_doc_idx += 1
            sequence_idx = 0
        else:
            prev_doc_idx = prev_doc_idxs[i]
            prev_sequence_idx = prev_sequence_idxs[i]

            if len(docs_segmented[prev_doc_idx]) - 1 == prev_sequence_idx:
                doc_idx = next_doc_idx
                next_doc_idx += 1
                sequence_idx = 0
                if doc_idx >= len(docs_segmented):
                    return None, None, None, None, None, None, None
            else:
                doc_idx = prev_doc_idx
                sequence_idx = prev_sequence_idx + 1
                if len(docs_segmented[doc_idx]) - 1 == sequence_idx:
                    is_last_sequence[i] = 1

        sequence, seq_bows = docs_segmented[doc_idx][sequence_idx]
        sequences.append(sequence)
        mask = np.ones(len(sequence))
        if sequence[-1] == cf.pad_idx:
            mask[(sequence == cf.pad_idx).argmax() :] = 0
        masks.append(mask)
        # if prev_doc_idxs is None or doc_idx != prev_doc_idx:
        #     prev_running_bows[i] = 0
        # if evaluate and not cf.eval_false:
        #     context = prev_running_bows[i].clone().detach()
        # elif not cf.use_all_bows:
        #     context = doc_bows[doc_idx] - seq_bows
        # else:
        #     context = doc_bows[doc_idx]
        # bows.append(context)
        # prev_running_bows[i] += seq_bows  # Add current bows only for the next batch
        cur_doc_idxs.append(doc_idx)
        cur_sequence_idxs.append(sequence_idx)

    batch = np.stack(sequences)
    mask = np.stack(masks)
    # bows = np.stack(bows)
    return (
        batch[:, :-1],
        # bows,
        None,
        batch[:, 1:],
        mask[:, 1:],
        cur_doc_idxs,
        cur_sequence_idxs,
        is_last_sequence,
    )
