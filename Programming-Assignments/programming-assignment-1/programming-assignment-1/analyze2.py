# analyze.py

import torch
import pickle

from model import Model
import glove_model as glove

def get_word_embedding(word, embedding_weights):
    assert word in data['vocab'], 'Word not in vocab'
    return embedding_weights[data['vocab'].index(word)]

# word4 = word1 - word2 + word3
def find_word_analogy(word1, word2, word3, embedding_weights):
    embedding1 = get_word_embedding(word1, embedding_weights)
    embedding2 = get_word_embedding(word2, embedding_weights)
    embedding3 = get_word_embedding(word3, embedding_weights)
    target_embedding = embedding1 - embedding2 + embedding3

    # Compute distance to every other word.
    diff = embedding_weights - target_embedding.reshape((1, -1))
    distance = torch.sqrt(torch.sum(diff ** 2, axis=1))

    # Sort by distance.
    order = torch.argsort(distance)[:10]
    print("The top 10 closest words to emb({}) - emb({}) + emb({}) are:".format(word1, word2, word3))
    for i in order:
        print('{}: {}'.format(data['vocab'][i], distance[i]))


torch.manual_seed(1)
n_epochs = 500  # A hyperparameter.  You can play with this if you want.
embedding_dim = 16

data_location = 'data.pk'
data = pickle.load(open(data_location, 'rb'), encoding='latin1')
vocab_size = len(data['vocab']) # Number of vocabs

W_final_sym, W_tilde_final_asym, W_final_asym = None, None, None
init_variance = 0.1  # A hyperparameter.  You can play with this if you want.
W = init_variance * torch.randn(size=(vocab_size, embedding_dim))
W_tilde = init_variance * torch.randn(size=(vocab_size, embedding_dim))
b = init_variance * torch.randn(size=(vocab_size, 1))
b_tilde = init_variance * torch.randn(size=(vocab_size, 1))

asym_log_co_occurence_train = glove.calculate_log_co_occurence(data['train_inputs'], vocab_size, symmetric=False)
asym_log_co_occurence_valid = glove.calculate_log_co_occurence(data['valid_inputs'], vocab_size, symmetric=False)

# Symmetric model
W_final_sym, _, b_final_sym, _ , _, _ = glove.train_GLoVE(W, None, b, None, asym_log_co_occurence_train, asym_log_co_occurence_valid, n_epochs)
# Asymmetric model
W_final_asym, W_tilde_final_asym, b_final_asym, b_tilde_final_asym, _, _ = glove.train_GLoVE(W, W_tilde, b, b_tilde, asym_log_co_occurence_train, asym_log_co_occurence_valid, n_epochs)

## GloVe embeddings
embedding_weights = W_final_sym # Symmetric GloVe
find_word_analogy('four', 'two', 'three', embedding_weights)

# Concatenation of W_final_asym, W_tilde_final_asym
embedding_weights = torch.cat((W_tilde_final_asym, W_final_asym), dim=1) 
find_word_analogy('four', 'two', 'three', embedding_weights)

# Averaging asymmetric GLoVE vectors
embedding_weights = (W_final_asym + W_tilde_final_asym)/2 
find_word_analogy('four', 'two', 'three', embedding_weights)

## Neural Netework Word Embeddings
args = {'batch_size': 100,                    # size of a mini-batch
        'learning_rate': 1e-3,                   # learning rate
        'momentum': 0.9,                      # decay parameter for the momentum vector
        'weight_decay': 0,                    # L2 regularization on the weights
        'epochs': 50,                         # maximum number of epochs to run
        'context_len': 4,                     # number of context words used
        'embedding_dim': 16,                  # number of dimensions in embedding
        'vocab_size': 251,                    # number of words in vocabulary
        'num_hid': 128,                       # number of hidden units
        'model_file': 'model.pk',             # filename to save best model
}
model = Model(args, data['vocab'])
model.load('model.pk')
weights = model.model.layers[0].weight # Neural network from part3
find_word_analogy('four', 'two', 'three', embedding_weights)
model.tsne_plot()
# model.display_nearest_words('he')