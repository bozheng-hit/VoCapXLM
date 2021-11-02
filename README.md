# VoCapXLM
Code for EMNLP2021 paper [Allocating Large Vocabulary Capacity for
Cross-lingual Language Model Pre-training](https://arxiv.org/pdf/2109.07306.pdf)

## Environment

DockerFile: `dancingsoul/pytorch:VoCapXLM`

Manully build the sentencepiece with following command:
```
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
```

## Data Preparation

1) Create a folder with `mkdir -p monolingual_text` in the root of this project.
2) Sample monolingual corpus for each language individually, move them to the monolingual_text directory, named after their language codes (e.g., en.txt). 
3) Sample the multilingual corpus from monolingual corpora with the following command:
```
python sample_multilingual_corpus.py \
    --lang_prob_path ./lang_prob_wiki.json \ 
    --input_dir ./monolingual_text/ \ 
    --output_path ./multilingual_corpus.text \
    --n_sample <n_sample> --beta <beta> --rescale
```
where the options are described as follows:
- `--lang_prob_path`: the probability of sampling training instances from each language during pre-training, `lang_prob_wiki.json` is counted on Wikipedia corpus and the probabilities are rescaled with alpha=0.7 from Equation (3) in our paper.
- `--n_sample`: number of sentences in the multilingual corpus where the final multilingual sentencepiece model is trained, the default value is 20000000. 
- `--rescale`: further rescale the probability with another value beta from Equation (2) in our paper. 
- `--beta`: the rescaling factor in Equation (2), the default value is 0.7. 

## Training Monolingual SentencePiece Models

Train monolingual sentencepiece models in different sizes to obtain vocabularies with different ALP, i.e., language-specific vocabulary capacity.
```
python train_mono_spm.py \
    --input_dir ./monolingual_text/ \
    --output_dir ~/monolingual_spm/ \
    --languages <all_languages> \
    --min_vocab_size <min_vocab_size> \
    --max_vocab_size <max_vocab_size> \
    --delta_vocab_size <delta_vocab_size> \
    --n_sample <n_sample>
```
where the options are described as follows:
- `--languages`: all languages under the monolingual_text directory, separated with `,`, e.g. `en,fr,zh`.
- `--min_vocab_size`: minimum vocabulary size allocated for each language, the default value is 1000.
- `--max_vocab_size`: maximum vocabulary size allocated for each language, the default value is 50000. 
- `--delta_vocab_size`: the value of interval to learn vocabularies, the default value is 1000.
- `--n_sample`: the number of sentences to calculate ALP for each language, the default value is 1000000.

or you can download our pre-trained monolingual sentencepiece models and vocabularies from [here][2].

## Allocating Multilingual Vocabulary

Allocate the multilingual vocabulary from monolingual vocabularies:
```
python train_vocap.py \
    --lang_prob_path ./lang_prob_wiki.json \
    --input_dir ./monolingual_spm/ \
    --output_path ./multilingual.vocab \
    --beta <beta> --rescale --target_vocab_size <target_vocab_size>
```
where the options are described as follows:
- `--lang_prob_path`: same as the above. 
- `--rescale`: same as the above. 
- `--beta`: same as the above.
- `--target_vocab_size`: the desired vocabulary size of the multilingual vocabulary, the default value is 500000.

Then Use sentencepiece to train the tokenizer given the multilingual vocabulary:
```
spm_train --input=./multilingual_corpus.text --model_prefix=<model_name> --vocab_size=<target_vocab_size> \
--character_coverage=0.9995 --model_type=unigram --shuffle_input_sentence=true \
--input_sentence_size=<input_sentence_size> --vocab_path=./multilingual.vocab
```
where the options are described as follows:
- `--model_prefix`: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
- `--character_coverage`: amount of characters covered by the model.
- `--vocab_size`: same as `--target_vocab_size`.
- `--vocab_path`: the required subwords in the final learned tokenizer.
## Paper
Please cite our paper `\cite{bo2021vocapxlm}` if you found the resources in the repository useful.

```
@inproceedings{bo2021vocapxlm,
author = {Bo Zheng, Li Dong, Shaohan Huang, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, Furu Wei},
booktitle = {Proceedings of EMNLP 2021},
title = {{Allocating Large Vocabulary Capacity for Cross-lingual Language Model Pre-training}},
year = {2021}
}
```

## Reference

1. https://github.com/google/sentencepiece
2. https://drive.google.com/file/d/1VttgE30xo-i1ig5xsMF_7R4AB2sA5J9F/view?usp=sharing