import json
import bcolz
import pickle

import numpy as np
import torch


class Vocabulary:
	def __init__(self):
		self.token_to_id = {"<PAD>": 0}
		self.id_to_token = {0: "<PAD>"}

	def build(self, tokens):
		for token in tokens:
			if token not in self.token_to_id:
				self.token_to_id[token] = len(self.token_to_id)
				self.id_to_token[self.token_to_id[token]] = token

	def __len__(self):
		return len(self.token_to_id)


class DataProcessor:
	def __init__(self, train_file, dev_file, test_file):
		self.max_seq_len = 0
		self.train_file = train_file
		self.dev_file = dev_file
		self.test_file = test_file
		self.type_to_label = {"Peop": "PER", "Org": "ORG", "Loc": "LOC", "Other": "O"}
		self.label_to_id = {"B-PER": 0, "I-PER": 1, "B-ORG": 2, "I-ORG": 3, "B-LOC": 4, "I-LOC": 5, "O": 6}

	def get_conll_examples(self, do_training=True):
		conll_examples = []
		if do_training is True:
			filename = [self.train_file, self.dev_file]
			file_num = 2
		else:
			filename = [self.test_file]
			file_num = 1

		vocab = Vocabulary()
		for i in range(file_num):
			examples = []
			print(filename[i])
			with open(filename[i], "r", encoding="utf-8") as reader:
				data = json.load(reader)
			reader.close()

			for data in data:
				tokens = data["tokens"]
				entities = data["entities"]
				relations = data["relations"]
				orig_id = data["orig_id"]

				vocab.build(tokens)
				token_len = len(tokens)
				if token_len > self.max_seq_len:
					self.max_seq_len = token_len

				label = [self.type_to_label["Other"]] * token_len
				for entity in entities:
					if entity["type"] == "Other":
						continue
					for idx in range(entity["start"], entity["end"]):
						if idx == entity["start"]:
							label[idx] = "B-" + self.type_to_label[entity["type"]]
						else:
							label[idx] = "I-" + self.type_to_label[entity["type"]]

				example = InputExample(
					orig_id=orig_id,
					tokens=tokens,
					label=label,
					relations=relations
				)
				examples.append(example)
			conll_examples.append(examples)
		return conll_examples, vocab

	def convert_example_to_features(self, examples, vocab):
		features = []
		labels = []
		for example in examples:
			ids = [vocab.token_to_id[tok] for tok in example.tokens]
			label = [self.label_to_id[lbl] for lbl in example.label]
			pad_len = self.max_seq_len - len(ids)
			ids.extend([0] * pad_len)
			label.extend([-1] * pad_len)
			features.append(ids)
			labels.append(label)

		features = torch.tensor(features, dtype=torch.long)
		labels = torch.tensor(labels, dtype=torch.long)
		return features, labels

	def build_lookup_matrix(self, vocab):
		vectors = bcolz.open('6B.50.dat')[:]
		words = pickle.load(open('6B.50_words.pkl', 'rb'))
		word2idx = pickle.load(open('6B.50_idx.pkl', 'rb'))

		glove = {w: vectors[word2idx[w]] for w in words}
		matrix_len = vocab.__len__()
		weights_matrix = np.zeros((matrix_len, 50))
		words_found = 0

		for i, word in vocab.id_to_token.items():
			try:
				weights_matrix[i] = glove[word]
				words_found += 1
			except KeyError:
				weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))
		return weights_matrix


class InputExample:
	def __init__(self,
				orig_id,
				tokens,
				label,
				relations):
		self.orig_id = orig_id
		self.tokens = tokens
		self.label = label
		self.relations = relations

