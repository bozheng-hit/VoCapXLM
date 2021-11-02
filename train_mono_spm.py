import argparse
import json
import glob
import re
import os
import math
import numpy as np
import sentencepiece as spm
import random


class Tokenizer(object):
    def __init__(self, vocab_file):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))

    def get_vocab(self):
        return [self.sp_model.IdToPiece(idx) for idx in range(len(self.sp_model))]

    def tokenize(self, text):
        return self.sp_model.EncodeAsIds(text)


def compute_alp(input_dir, lang, vocab_file, n_sample):
    random.seed(1)
    input_file = os.path.join(input_dir, "{}.txt".format(lang))
    with open(input_file, "r") as fin:
        lines = fin.readlines()

    all_tokens = 0
    tokenizer = Tokenizer(vocab_file)
    words_list = tokenizer.get_vocab()
    words = {}
    for i, word in enumerate(words_list):
        words[i] = 0

    random.shuffle(lines)
    line_idx = 0
    tokenized_lines = []
    for line in lines[:n_sample]:
        line_idx += 1
        if line_idx % 100000 == 0:
            print("tokenized {} lines.".format(line_idx))
        line = line.strip()
        token_ids = tokenizer.tokenize(line)
        all_tokens += len(token_ids)
        for idx in token_ids:
            words[idx] += 1
        tokenized_lines.append(token_ids)
    for idx in words.keys():
        words[idx] /= all_tokens
    probs = []
    for token_ids in tokenized_lines:
        p = 0.0
        for idx in token_ids:
            p += math.log(words[idx])
        probs.append(p)

    return np.mean(probs)


def train_spm(input_dir, output_dir, lang, vocab_size, vocab_path=None):
    random.seed(1)
    output_dir = os.path.join(output_dir, lang)
    os.makedirs(output_dir, exist_ok=True)
    input_file = os.path.join(input_dir, "{}.txt".format(lang))
    model_prefix = os.path.join(output_dir, "{}.{}".format(lang, vocab_size))
    if not os.path.exists(model_prefix + ".model"):

        try:
            spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix, vocab_size=vocab_size,
                                           character_coverage=0.9995, model_type="unigram", shuffle_input_sentence=True,
                                           input_sentence_size=1000000)
            # cmd = "spm_train --input={} --model_prefix={} --vocab_size={} --character_coverage= 0.9995 --model_type=unigram --shuffle_input_sentence=true --input_sentence_size=1000000 --vocab_path={}".format(
            #     input_file, model_prefix, vocab_size, vocab_path)
            # os.system(cmd)
        except:
            return None
    return model_prefix + ".model"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="path to raw text file.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="output_dir",
    )
    parser.add_argument(
        "--vocab_path",
        default=None,
        type=str,
        required=False,
        help="sentencepiece model path",
    )
    parser.add_argument(
        "--languages",
        default=None,
        type=str,
        required=True,
        help="languages",
    )
    parser.add_argument(
        "--min_vocab_size",
        default=1000,
        type=int,
        required=True,
        help="min vocab size",
    )
    parser.add_argument(
        "--max_vocab_size",
        default=50000,
        type=int,
        required=True,
        help="max vocab size",
    )
    parser.add_argument(
        "--delta_vocab_size",
        default=1000,
        type=int,
        required=True,
        help="delta vocab size",
    )
    parser.add_argument(
        "--n_sample",
        default=1000000,
        type=int,
        required=True,
        help="lines to sample in each file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert args.max_vocab_size >= args.min_vocab_size
    assert (args.max_vocab_size - args.min_vocab_size) % args.delta_vocab_size == 0

    languages = args.languages.split(',')
    # lang = "zh"
    for lang in languages:
        alp_vocab = []
        for vocab_size in range(args.min_vocab_size, args.max_vocab_size + 1, args.delta_vocab_size):
            vocab_file = train_spm(args.input_dir, args.output_dir, lang, vocab_size, args.vocab_path)
            if vocab_file is None:
                continue
            alp = compute_alp(args.input_dir, lang, vocab_file, args.n_sample)
            alp_vocab.append([alp, vocab_size])
            print("language: {}, ALP: {}, vocab_size: {}".format(lang, alp, vocab_size))
        log_output_path = os.path.join(args.output_dir, lang, "{}.log".format(lang))
        fout = open(log_output_path, "w")
        fout.write(str(alp_vocab))
        fout.close()
