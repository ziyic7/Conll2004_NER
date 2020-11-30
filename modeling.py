import torch
from torch import nn
import torch.nn.functional as F


def argmax(vec):
	_, idx = torch.max(vec, 1)
	return idx.item()


def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


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
	def __init__(self, weight_matrix, hidden_dim, label_to_id, start_tag="<START>", stop_tag="<STOP>"):
		super(BiLSTM_CRF, self).__init__()
		self.START_TAG = start_tag
		self.STOP_TAG = stop_tag
		self.embedding, self.vocab_size, self.embedding_dim = self.init_embed_layer(weight_matrix)
		self.hidden_dim = hidden_dim
		self.label_to_id = label_to_id
		self.tagset_size = len(label_to_id)

		self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
		self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=1, bidirectional=True)

		self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

		self.transitions = nn.Parameter(
			torch.randn(self.tagset_size, self.tagset_size))

		self.transitions.data[label_to_id[start_tag], :] = -10000
		self.transitions.data[:, label_to_id[stop_tag]] = -10000

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

	def _forward_alg(self, feats, mask):
		init_alphas = torch.full((feats.size(0), self.tagset_size), -10000.)
		init_alphas[:, self.label_to_id[self.START_TAG]] = 0.

		forward_var_list = []
		forward_var_list.append(init_alphas)
		d = torch.unsqueeze(feats[:, 0], dim=1)
		for feat_index in range(1, feats.size(1)):
			n_unfinish = mask[:, feat_index].sum()
			if n_unfinish.item() == 0:
				break
			d_uf = d[:n_unfinish]
			emit_and_transition = feats[: n_unfinish, feat_index].unsqueeze(dim=1) + self.transitions
			log_sum = d_uf.transpose(1, 2) + emit_and_transition
			max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
			log_sum = log_sum - max_v
			d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
			d = torch.cat((d_uf, d[n_unfinish:]), dim=0)
		d = d.squeeze(dim=1)
		max_d = d.max(dim=-1)[0]
		d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
		return d

	def _get_lstm_features(self, sentences):
		embeds = self.word_embeds(sentences).permute(1, 0, 2)
		lstm_out, self.hidden = self.lstm(embeds)
		lstm_out = lstm_out.permute(1, 0, 2)
		lstm_feats = self.hidden2tag(lstm_out)
		return lstm_feats

	def _score_sentence(self, feats, tags, mask):

		score = torch.gather(feats, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2).to("cuda")
		score[:, 1:] += self.transitions[tags[:, :-1], tags[:, 1:]]
		total_score = (score * mask.type(torch.float)).sum(dim=1)
		return total_score

	def _viterbi_decode(self, feats, mask):
		batch_size = feats.size(0)
		tags = [[[i] for i in range(self.tagset_size)]] * batch_size
		d = torch.unsqueeze(feats[:, 0], dim=1)
		for i in range(1, feats.size(1)):
			n_unfinished = mask[:, i].sum()
			if n_unfinished.item() == 0:
				break
			d_uf = d[: n_unfinished]
			emit_and_transition = self.transitions + feats[: n_unfinished, i].unsqueeze(dim=1)
			new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
			d_uf, max_idx = torch.max(new_d_uf, dim=1)
			max_idx = max_idx.tolist()
			tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
			d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)
		d = d.squeeze(dim=1)
		score, max_idx = torch.max(d, dim=1)
		max_idx = max_idx.tolist()
		tags = [tags[b][k] for b, k in enumerate(max_idx)]
		return score, tags

	def neg_log_likelihood(self, sentences, tags):
		mask = (tags > 0).to("cuda")
		feats = self._get_lstm_features(sentences)
		forward_score = self._forward_alg(feats, mask)
		gold_score = self._score_sentence(feats, tags, mask)
		return forward_score - gold_score

	def forward(self, sentences, tags):
		mask = (tags > 0).to("cuda")
		lstm_feats = self._get_lstm_features(sentences)
		score, tag_seq = self._viterbi_decode(lstm_feats, mask)
		return score, tag_seq
