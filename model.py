import torch.nn.functional as F
from torch import nn


class CNNModel(nn.Module):
    def __init__(self, embedding_dim, kernel_size, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self._kernel_size = kernel_size

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.conv = nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size, embedding_dim))

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sample):
        embeds = self.word_embeddings(sample)
        conv_in = embeds.view(1, 1, len(sample), -1)
        conv_out = self.conv(conv_in)
        conv_out = F.relu(conv_out)
        hidden_in = conv_out.view(self.hidden_dim, len(sample) + 1 - self._kernel_size).transpose(0, 1)
        tag_space = self.hidden2tag(hidden_in)
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores
