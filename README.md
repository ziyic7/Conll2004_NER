# NER_BiLSTM_CRF

Name entity recognition on conll04 dataset

## Requirements

Pytorch 1.6.0

bcolz 1.2.0

## Load Glove vectors

```shell
python prepare_vectors.py
```

## Run BiLSTM

```shell
python run_ner.py --do_eval
```

## Run BiLSTM_CRF

```sh
python run_ner.py --do_eval --with_crf
```

