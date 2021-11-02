import argparse
import json
import glob
import re
import os
import math
import numpy as np
import sentencepiece as spm
import random
import ast
from collections import OrderedDict


def merge_vocab(input_dir, lang_alp_vocab, threshold, target_vocab_size):
    vocab = OrderedDict()
    lang_cnt = {}

    language_need_larger_vocab = []

    for lang in lang_alp_vocab:
        ptr = None
        alp_vocab = lang_alp_vocab[lang]
        for i in range(1, len(alp_vocab)):
            if alp_vocab[i][0] - alp_vocab[i - 1][0] > threshold:
                ptr = i
        if ptr is not None and ptr + 1 == len(alp_vocab):
            language_need_larger_vocab.append(lang)

        if ptr is not None:
            lines = open(os.path.join(input_dir, lang, "{}.{}.vocab".format(lang, alp_vocab[ptr][1])), "r").readlines()
            for line in lines:
                word = line.strip().split("\t")[0]
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
            lang_cnt[lang] = alp_vocab[ptr][1] - alp_vocab[0][1]
        else:
            lang_cnt[lang] = 0

    if len(language_need_larger_vocab) > 0 and len(vocab) <= target_vocab_size:
        print("These languages need bigger vocabularies: {}".format(language_need_larger_vocab))
        assert False

    print(lang_cnt, len(vocab))

    return vocab, lang_cnt


def pad_vocab(vocab, input_path, target_size):
    lines = open(input_path, "r").readlines()
    for line in lines:
        if len(vocab) >= target_size:
            return vocab
        else:
            word = line.strip().split("\t")[0]
            if word not in vocab:
                vocab[word] = 1
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--lang_prob_path",
        default=None,
        type=str,
        required=True,
        help="path to meta file.",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="path to raw text file.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="output_dir",
    )
    parser.add_argument(
        "--beta",
        default=0.7,
        type=float,
        required=False,
        help="beta",
    )
    parser.add_argument(
        "--rescale",
        action="store_true"
    )
    parser.add_argument(
        "--target_vocab_size",
        default=500000,
        type=int,
        required=True,
        help="min vocab size",
    )

    args = parser.parse_args()

    lang_prob_dict = json.loads(open(args.lang_prob_path, "r").readlines()[0])

    z = sum([lang_prob_dict[lang] for lang in lang_prob_dict])

    for lang in sorted(lang_prob_dict.keys()):
        lang_prob_dict[lang] = lang_prob_dict[lang] / z

    print(lang_prob_dict)

    if args.beta != 1:
        print("renorm lang_prob with beta = {}.".format(args.beta))
        z = sum([math.pow(lang_prob_dict[lang], args.beta) for lang in lang_prob_dict])
        for lang in sorted(lang_prob_dict.keys()):
            lang_prob_dict[lang] = math.pow(lang_prob_dict[lang], args.beta) / z

    languages = []

    lang_alp_vocab = {}

    for lang in os.listdir(args.input_dir):
        if lang not in lang_prob_dict:
            print("language {} not found in lang_prob_dict.".format(lang))
            continue
        output_dir = os.path.join(args.input_dir, lang)
        log_path = os.path.join(output_dir, "{}.log".format(lang))
        s = open(log_path, "r").readlines()[0]
        alp_vocab = ast.literal_eval(s)

        # smooth alp / vocab curve such that its increment is monotonically decreasing
        for i in range(2, len(alp_vocab)):
            j = i
            while j >= 2:
                if alp_vocab[j][0] - alp_vocab[j - 1][0] > alp_vocab[j - 1][0] - alp_vocab[j - 2][0]:
                    kv = (alp_vocab[i][0] - alp_vocab[j - 2][0]) / (i - j + 2)
                    mv = (i + j - 2) / 2
                    for k in range(j - 2, i):
                        alp_vocab[k + 1][0] = alp_vocab[k][0] + kv + (mv - k) * 1e-6
                    j -= 1
                else:
                    break

        languages.append(lang)

        for i in range(2, len(alp_vocab)):
            if alp_vocab[i][0] - alp_vocab[i - 1][0] > alp_vocab[i - 1][0] - alp_vocab[i - 2][0]:
                assert False

        if args.rescale:
            for i in range(len(alp_vocab)):
                alp_vocab[i][0] *= lang_prob_dict[lang]

        lang_alp_vocab[lang] = alp_vocab

    for lang in lang_alp_vocab:
        print(lang, len(lang_alp_vocab[lang]))


    # make sure each language has language-specific subword units.
    for lang in lang_alp_vocab:
        lang_alp_vocab[lang] = [[-10000, 0]] + lang_alp_vocab[lang]

    left = 0
    right = 100
    while left + 1e-6 < right:
        mid = (left + right) / 2
        vocab, _ = merge_vocab(args.input_dir, lang_alp_vocab, mid, args.target_vocab_size)
        if len(vocab) <= args.target_vocab_size:
            right = mid
        else:
            left = mid

    vocab, lang_cnt = merge_vocab(args.input_dir, lang_alp_vocab, right, args.target_vocab_size)

    vocab = pad_vocab(vocab, os.path.join(args.input_dir, "en", "en.{}.vocab".format(lang_alp_vocab["en"][-1][1])),
                      args.target_vocab_size)

    with open(args.output_path, "w") as fout:
        for word in vocab:
            fout.write(word + '\n')
