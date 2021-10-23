# model.py

import math
import torch
import pickle
import layer as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Model:
    def __init__(self, args, vocab):
        self.model = nn.Sequential()
        self.model.add_layer(nn.WordEmbedding(args['vocab_size'], args['embedding_dim']))
        self.model.add_layer(nn.Linear(args['embedding_dim'] * args['context_len'], args['num_hid']))
        self.model.add_layer(nn.Sigmoid())
        self.model.add_layer(nn.Linear(args['num_hid'], args['context_len'] * args['vocab_size']))
        self.model.add_layer(nn.SoftMax(args['context_len']))
        self.criterion = nn.CrossEntropy(args['vocab_size'])
        self.vocab = vocab

    def save(self, name):
        f = open(name, 'wb')
        pickle.dump(self, f)
        f.close()

    def load(self, name):
        f = open(name, 'rb')
        temp = pickle.load(f, encoding='latin1')
        f.close()
        self.model = temp.model
        self.criterion = temp.criterion
        self.vocab = temp.vocab

    def tsne_plot(self):
        """2-D visualization of the learned representations using t-SNE."""
        mapped_X = TSNE(2).fit_transform((self.model.layers[0].weight.numpy()))
        plt.figure()
        for i, w in enumerate(self.vocab):
            plt.text(mapped_X[i, 0], mapped_X[i, 1], w)
        plt.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
        plt.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
        plt.show()

    def word_distance(self, word1, word2):
        """Distance between vector representations of two words."""

        if word1 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word1))
        if word2 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word2))

        idx1, idx2 = self.vocab.index(word1), self.vocab.index(word2)
        word_rep1 = self.model.layers[0].weight[idx1, :]
        word_rep2 = self.model.layers[0].weight[idx2, :]
        return (word_rep1 - word_rep2).norm().item()

    def predict_next_word(self, word1, word2, word3, k=10):
        """List the top k predictions for the next word along with their probabilities.
        Inputs:
            word1: The first word as a string.
            word2: The second word as a string.
            word3: The third word as a string.
            k: The k most probable predictions are shown.
        Example usage:
            model.predict_next_word('john', 'might', 'be', 3)
            model.predict_next_word('life', 'in', 'new', 3)"""

        if word1 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word1))
        if word2 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word2))
        if word3 not in self.vocab:
            raise RuntimeError('Word "{}" not in vocabulary.'.format(word3))

        idx1 = self.vocab.index(word1)
        idx2 = self.vocab.index(word2)
        idx3 = self.vocab.index(word3)
        input = torch.Tensor([idx1, idx2, idx3]).reshape((1, 3)).long()
        prob = self.model.forward(input)
        _, idxs = prob.sort(dim=1, descending=True)   # sort descending
        idxs = idxs.squeeze()

        for i in range(k):
            print('{} {} {} {} Prob: {:1.5f}'.format(word1, word2, word3, self.vocab[idxs[i]], prob[0, idxs[i]]))

    def display_nearest_words(self, word, k=10):
        """Find k nearest words to a given word, along with their distances."""
        if word not in self.vocab:
            print('Word "{}" not in vocabulary.'.format(word))
            return

        # Compute distance to every other word.
        idx = self.vocab.index(word)
        word_rep = self.model.layers[0].weight[idx, :]
        diff = torch.add(self.model.layers[0].weight, - word_rep)
        distance = diff.norm(p=2, dim=1)

        # Sort by distance.
        _, order = distance.sort()
        order = order[1:1 + k]  # skip nearest word (query itself)
        for i in order:
            print('{}: {}'.format(self.vocab[i], distance[i]))
    
    def evaluate(self, dataloader, batch_size):
        total_loss = 0
        for _ in range(math.ceil(dataloader.get_size() / batch_size)):
            self.model.zero_grad()
            input = dataloader.get_batch()
            mask = dataloader.sample_mask(batch_size)
            input_masked = input * (1 - mask)
            target_masked = input * mask
            output = self.model.forward(input_masked)
            loss = self.criterion.forward(output, target_masked)
            total_loss += loss
        total_loss = total_loss / dataloader.get_size()
        print("Evaluation: Loss:  %.4f \t" % (total_loss))
        print(' ')
        return total_loss.item()