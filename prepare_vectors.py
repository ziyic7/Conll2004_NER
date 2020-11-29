import bcolz
import pickle

import numpy as np


def load_glove_vectors(filename):
	vocab = []
	idx = 0
	word2idx = {}
	vectors = bcolz.carray(np.zeros(1), rootdir="6B.50.dat", mode="w")

	with open(filename, "rb") as f:
		for l in f:
			line = l.decode().split()
			word = line[0]
			vocab.append(word)
			word2idx[word] = idx
			idx += 1
			vector = np.array(line[1:]).astype(np.float)
			vectors.append(vector)

	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir="6B.50.dat", mode="w")
	vectors.flush()
	pickle.dump(vocab, open("6B.50_words.pkl", "wb"))
	pickle.dump(word2idx, open("6B.50_idx.pkl", "wb"))


if __name__ == "__main__":
	load_glove_vectors("glove.6B.50d.txt")