
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
# find_word_analogy('he', 'him', 'her', weights)
# model.tsne_plot()
model.display_nearest_words('she')
print(model.word_distance('he','she'))