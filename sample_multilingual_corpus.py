import argparse
import json
import glob
import re
import os
import math
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--lang_prob_path",
        default=None,
        type=str,
        required=True,
        help="path to lang prob file",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="input_dir",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="output_path.",
    )
    parser.add_argument(
        "--n_sample",
        default=20000000,
        type=int,
        required=True,
        help="lines to sample.",
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
    args = parser.parse_args()

    lang_prob_dict = json.loads(open(args.lang_prob_path, "r").readlines()[0])

    z = sum([lang_prob_dict[lang] for lang in lang_prob_dict])

    for lang in sorted(lang_prob_dict.keys()):
        lang_prob_dict[lang] = lang_prob_dict[lang] / z

    print(sum([lang_prob_dict[lang] for lang in lang_prob_dict]))

    print(lang_prob_dict)

    if args.rescale and args.beta != 1:
        print("renorm lang_prob with beta = {}.".format(args.beta))
        z = sum([math.pow(lang_prob_dict[lang], args.beta) for lang in lang_prob_dict])
        for lang in sorted(lang_prob_dict.keys()):
            lang_prob_dict[lang] = math.pow(lang_prob_dict[lang], args.beta) / z

    n_sample = {}

    for lang in lang_prob_dict:
        n_sample[lang] = int(lang_prob_dict[lang] * args.n_sample)

    random.seed(1)

    with open(args.output_path, "w") as fout:
        for file in os.listdir(args.input_dir):
            input_path = os.path.join(args.input_dir, file)
            if not os.path.exists(input_path):
                print("{} does not exist.".format(input_path))
                continue
            else:
                print("processing {}.".format(file))

            lang = file.split(".")[0]
            if lang not in n_sample:
                print("skipping language {}.".format(lang))
                continue

            print(lang, n_sample[lang])

            lines = open(input_path, "r").readlines()
            random.shuffle(lines)
            assert len(lines) >= n_sample[lang]
            for line in lines[:n_sample[lang]]:
                fout.write(line.strip() + '\n')
