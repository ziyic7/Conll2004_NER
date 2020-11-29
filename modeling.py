import torch
from torch import nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
	def __init__(self, weight_matrix, hidden_size, number_of_tags):
		super(BiLSTM, self).__init__()
		self.embedding, vocab_size, embedding_dim = self.init_embed_layer(weight_matrix)
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, bidirectional=True)
		self.hidden_to_tag = nn.Linear(2 * hidden_size, number_of_tags)  # double the hidden size because bi-directional

	def init_embed_layer(self, weight_matrix, do_training=False):
		"""
		:param weight_matrix: matrix compacted by pre-trained glove vectors
		:param do_training: to train the word vectors
		"""
		vocab_size, embedding_dim = weight_matrix.shape
		embedding_layer = nn.Embedding(vocab_size, embedding_dim)
		embedding_layer.from_pretrained(torch.FloatTensor(weight_matrix))  # load the pre-trained
		if not do_training:
			embedding_layer.weight.requires_grad = False
		return embedding_layer, vocab_size, embedding_dim

	def forward(self, x):
		self.lstm.flatten_parameters()
		embedding = self.embedding(x).permute(1, 0, 2)
		output, _ = self.lstm(embedding)  # (seq_len, batch_sz, embedding_dim)->(seq_len, batch_sz, 2 * hidden_sz)
		output = output.view(-1, output.shape[2])  # ->(batch_sz * seq_len, 2 * hidden_sz)
		logits = self.hidden_to_tag(output)  # ->(batch_sz * seq_len, num_of_tags)
		return F.log_softmax(logits, dim=1)


class BiLSTM_CRF(nn.Module):
	def __init__(self):
		super(BiLSTM_CRF, self).__init__()



