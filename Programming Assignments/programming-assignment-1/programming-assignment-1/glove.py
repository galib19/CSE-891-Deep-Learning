# glove.py

import torch
import pickle
import numpy as np
from tqdm import tqdm
# import plot
import matplotlib.pylab as plot
from sklearn.manifold import TSNE

import glove_model as glove

TINY = 1e-30
EPS = 1e-4
nax = np.newaxis
torch.set_default_dtype(torch.float64)

data_location = 'data.pk'
data = pickle.load(open(data_location, 'rb'), encoding='latin1')
print(data['vocab'][0]) # First word in vocab is [MASK] 
print(data['vocab'][1]) 
print(len(data['vocab'])) # Number of words in vocab
print(data['vocab']) # All the words in vocab
print(data['train_inputs'][:10]) # 10 example training instances
vocab_size = len(data['vocab']) # Number of vocabs

asym_log_co_occurence_train = glove.calculate_log_co_occurence(data['train_inputs'], vocab_size, symmetric=False)
asym_log_co_occurence_valid = glove.calculate_log_co_occurence(data['valid_inputs'], vocab_size, symmetric=False)

"""
Train the GLoVE model for a range of embedding dimensions
"""

torch.manual_seed(1)
n_epochs = 500  # A hyperparameter.  You can play with this if you want.
embedding_dims = [1, 2, 10, 128, 256]  # Play with this
# Store the final losses for graphing
asymModel_asymCoOc_final_train_losses, asymModel_asymCoOc_final_val_losses = [], []
symModel_asymCoOc_final_train_losses, symModel_asymCoOc_final_val_losses = [], []
Asym_W_final_2d, Asym_b_final_2d, Asym_W_tilde_final_2d, Asym_b_tilde_final_2d = None, None, None, None
W_final_2d, b_final_2d = None, None
do_print = True  # If you want to see diagnostic information during training

for embedding_dim in tqdm(embedding_dims):
  init_variance = 0.1  # A hyperparameter.  You can play with this if you want.
  W = init_variance * torch.randn(vocab_size, embedding_dim)
  W_tilde = init_variance * torch.randn(vocab_size, embedding_dim)
  b = init_variance * torch.randn(vocab_size, 1)
  b_tilde = init_variance * torch.randn(vocab_size, 1)
  if do_print:
    print(f"Training for embedding dimension: {embedding_dim}")
  
  # Train Asym model on Asym Co-Oc matrix
  Asym_W_final, Asym_W_tilde_final, Asym_b_final, Asym_b_tilde_final, train_loss, valid_loss = glove.train_GLoVE(W, W_tilde, b, b_tilde, asym_log_co_occurence_train, asym_log_co_occurence_valid, n_epochs, do_print=do_print)
  if embedding_dim == 2:
    # Save a parameter copy if we are training 2d embedding for visualization later
    Asym_W_final_2d = Asym_W_final
    Asym_W_tilde_final_2d = Asym_W_tilde_final
    Asym_b_final_2d = Asym_b_final
    Asym_b_tilde_final_2d = Asym_b_tilde_final
  asymModel_asymCoOc_final_train_losses += [train_loss]
  asymModel_asymCoOc_final_val_losses += [valid_loss]
  if do_print:
    print(f"Final validation loss: {valid_loss}")
  
  # Train Sym model on Asym Co-Oc matrix
  W_final, W_tilde_final, b_final, b_tilde_final, train_loss, valid_loss = glove.train_GLoVE(W, None, b, None, asym_log_co_occurence_train, asym_log_co_occurence_valid, n_epochs, do_print=do_print)
  if embedding_dim == 2:
    # Save a parameter copy if we are training 2d embedding for visualization later
    W_final_2d = W_final
    b_final_2d = b_final
  symModel_asymCoOc_final_train_losses += [train_loss]
  symModel_asymCoOc_final_val_losses += [valid_loss]
  if do_print:
    print(f"Final validation loss: {valid_loss}")

"""Plot the training and validation losses against the embedding dimension."""

plot.loglog(embedding_dims, asymModel_asymCoOc_final_train_losses, label="Asymmetric Model / Asymmetric Co-Oc", linestyle="--")
plot.loglog(embedding_dims, symModel_asymCoOc_final_train_losses , label="Symmetric Model / Asymmetric Co-Oc")
plot.xlabel("Embedding Dimension")
plot.ylabel("Training Loss")
plot.legend()
plot.show()

plot.loglog(embedding_dims, asymModel_asymCoOc_final_val_losses, label="Asymmetric Model / Asymmetric Co-Oc", linestyle="--")
plot.loglog(embedding_dims, symModel_asymCoOc_final_val_losses , label="Sym Model / Asymmetric Co-Oc")
plot.xlabel("Embedding Dimension")
plot.ylabel("Validation Loss")
plot.legend(loc="upper left")
plot.show()

glove.tsne_plot_GLoVE_representation(W_final, b_final, data_location)
glove.plot_2d_GLoVE_representation(W_final_2d, b_final_2d, data_location)
glove.tsne_plot_GLoVE_representation(W_final_2d, b_final_2d, data_location)