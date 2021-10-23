# glove.py

import torch
import pickle
import numpy as np
from tqdm import tqdm
import pylab
from sklearn.manifold import TSNE

TINY = 1e-30
EPS = 1e-4
nax = np.newaxis
torch.set_default_dtype(torch.float64)

def calculate_log_co_occurence(word_data, vocab_size, symmetric=False):
  "Compute the log-co-occurence matrix for our data."
  log_co_occurence = torch.zeros((vocab_size, vocab_size))
  for input in word_data:
    # Note: the co-occurence matrix may not be symmetric
    log_co_occurence[input[0], input[1]] += 1
    log_co_occurence[input[1], input[2]] += 1
    log_co_occurence[input[2], input[3]] += 1
    # If we want symmetric co-occurence can also increment for these.
    if symmetric:
      log_co_occurence[input[1], input[0]] += 1
      log_co_occurence[input[2], input[1]] += 1
      log_co_occurence[input[3], input[2]] += 1
  delta_smoothing = 0.5  # A hyperparameter.  You can play with this if you want.
  log_co_occurence += delta_smoothing  # Add delta so log doesn't break on 0's.
  log_co_occurence = torch.log(log_co_occurence)
  return log_co_occurence

def loss_GLoVE(W, W_tilde, b, b_tilde, log_co_occurence):
  "Compute the GLoVE loss."
  n,_ = log_co_occurence.shape
  if W_tilde is None and b_tilde is None:
    return torch.sum((W @ W.T + b @ torch.ones([1,n]) + torch.ones([n,1])@b.T - log_co_occurence)**2)
  else:
    return torch.sum((W @ W_tilde.T + b @ torch.ones([1,n]) + torch.ones([n,1])@b_tilde.T - log_co_occurence)**2)

def grad_GLoVE(W, W_tilde, b, b_tilde, log_co_occurence):
  "Return the gradient of GLoVE objective w.r.t W and b."
  "INPUT: W - Vxd; W_tilde - Vxd; b - Vx1; b_tilde - Vx1; log_co_occurence: VxV"
  "OUTPUT: grad_W - Vxd; grad_W_tilde - Vxd, grad_b - Vx1, grad_b_tilde - Vx1"
  n,_ = log_co_occurence.shape
  
  if not W_tilde is None and not b_tilde is None:
      loss = (W @ W_tilde.T + b @ torch.ones([1, n]) + torch.ones([n, 1]) @ b_tilde.T - 0.5 * (log_co_occurence + log_co_occurence.T))
      grad_W = 2 * (W_tilde.T @ loss).T
      grad_W_tilde = 2 * (W.T @ loss).T
      grad_b = 2 * (torch.ones([1, n]) @ loss).T
      grad_b_tilde = 2 * (torch.ones([n, 1]).T  @ loss).T
  else:
    loss = (W @ W.T + b @ torch.ones([1,n]) + torch.ones([n,1])@b.T - 0.5*(log_co_occurence + log_co_occurence.T))
    grad_W = 4 *(W.T @ loss).T
    grad_W_tilde = None
    grad_b = 4 * (torch.ones([1,n]) @ loss).T
    grad_b_tilde = None
  
  return grad_W, grad_W_tilde, grad_b, grad_b_tilde

def train_GLoVE(W, W_tilde, b, b_tilde, log_co_occurence_train, log_co_occurence_valid, n_epochs, do_print=False):
  "Traing W and b according to GLoVE objective."
  n,_ = log_co_occurence_train.shape
  learning_rate = 0.05 / n  # A hyperparameter.  You can play with this if you want.
  for epoch in range(n_epochs):
    grad_W, grad_W_tilde, grad_b, grad_b_tilde = grad_GLoVE(W, W_tilde, b, b_tilde, log_co_occurence_train)
    W = W - learning_rate * grad_W
    b = b - learning_rate * grad_b
    if not grad_W_tilde is None and not grad_b_tilde is None:
      W_tilde = W_tilde - learning_rate * grad_W_tilde
      b_tilde = b_tilde - learning_rate * grad_b_tilde
    train_loss, valid_loss = loss_GLoVE(W, W_tilde, b, b_tilde, log_co_occurence_train), loss_GLoVE(W, W_tilde, b, b_tilde, log_co_occurence_valid)
    if do_print:
      print(f"Train Loss: {train_loss}, valid loss: {valid_loss}, grad_norm: {torch.sum(grad_W**2)}")
  return W, W_tilde, b, b_tilde, train_loss, valid_loss

def tsne_plot_GLoVE_representation(W_final, b_final, data_location):
    """Plot a 2-D visualization of the learned representations using t-SNE."""
    mapped_X = TSNE(n_components=2).fit_transform(W_final)
    pylab.figure(figsize=(12,12))
    data_obj = pickle.load(open(data_location, 'rb'))
    for i, w in enumerate(data_obj['vocab']):
        pylab.text(mapped_X[i, 0], mapped_X[i, 1], w)
    pylab.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
    pylab.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
    pylab.show()

def plot_2d_GLoVE_representation(W_final, b_final, data_location):
    """Plot a 2-D visualization of the learned representations."""
    mapped_X = W_final
    pylab.figure(figsize=(12,12))
    data_obj = pickle.load(open(data_location, 'rb'))
    for i, w in enumerate(data_obj['vocab']):
        pylab.text(mapped_X[i, 0], mapped_X[i, 1], w)
    pylab.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
    pylab.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
    pylab.show()