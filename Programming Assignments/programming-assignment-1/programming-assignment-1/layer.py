# layer.py

import math
import torch

TINY = 1e-30


class Sequential:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
        return input

    def backward(self, grad_output):
        # print("seq", grad_output.shape)
        for i in range(len(self.layers) - 1, -1, -1):
            grad_output = self.layers[i].backward(grad_output)
        return grad_output

    def zero_grad(self):
        for i in range(len(self.layers)):
            self.layers[i].zero_grad()


class WordEmbedding:
    def __init__(self, vocab_size, embedding_dim, weight=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bias = torch.zeros(0)
        if weight is not None:
            self.weight = weight
        else:
            self.weight = torch.randn(vocab_size, embedding_dim)
        self.grad_bias = torch.zeros(0)
        self.grad_weight = torch.zeros(vocab_size, embedding_dim)
        self.is_trainable = True

    def get_indicator_matrix(self, indices):
        batch_size = indices.size(0)
        self.indicator_matrix = torch.zeros(batch_size, self.vocab_size)
        for i in range(batch_size):
            self.indicator_matrix[i, indices[i]] = 1.0

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.size(0)
        output = torch.zeros(self.input.size(0), self.input.size(1) * self.embedding_dim)
        for i in range(self.input.size(1)):
            self.get_indicator_matrix(self.input[:, i].long())
            output[:, i * self.embedding_dim:(i + 1) * self.embedding_dim] = \
                torch.mm(self.indicator_matrix, self.weight)
        return output

    def backward(self, grad_output):
        for i in range(self.input.size(1)):
            self.get_indicator_matrix(self.input[:, i].long())
            self.grad_weight = self.grad_weight + self.indicator_matrix.T.matmul(grad_output[:, i * self.embedding_dim:(i + 1) * self.embedding_dim])

    def zero_grad(self):
        self.grad_weight.fill_(0.0)


class Linear:
    def __init__(self, nin, nout, weight=None, bias=None):

        init_wt = math.sqrt(1 / nin)

        if bias is not None:
            self.bias = bias
        else:
            self.bias = (2 * torch.rand(nout) - 1) * init_wt

        if weight is not None:
            self.weight = weight
        else:
            self.weight = (2 * torch.rand(nin, nout) - 1) * init_wt
        
        self.grad_bias = torch.zeros(nout)
        self.grad_weight = torch.zeros(nin, nout)
        self.is_trainable = True
        # print("linear in out:", nin, nout)

    def forward(self, input):
        # print(input.shape)
        self.input = input
        output = torch.mm(input, self.weight)
        output = output.add(self.bias)
        return output

    def backward(self, grad_output):
        # print("linear backward", self.input.shape, grad_output.shape, grad_output.view(self.input.size(0), -1).matmul(self.weight.T).shape)
        self.grad_weight = torch.matmul(self.input.T, grad_output.view(self.input.size(0), -1))
        self.grad_bias = torch.sum(grad_output.view(self.input.size(0), -1), axis=0)
        return torch.matmul(grad_output.view(self.input.size(0), -1),(self.weight.T))

    def zero_grad(self):
        self.grad_bias.fill_(0.0)
        self.grad_weight.fill_(0.0)


class Sigmoid():

    def __init__(self):
        self.is_trainable = False

    def forward(self, input):
        self.input = input
        temp = torch.exp(-input)
        output = torch.div(torch.ones(temp.size()), torch.add(temp, 1.0))
        self.input = output
        return output

    def backward(self, grad_output):
        gradient = torch.mul(self.input, torch.add(torch.ones(self.input.shape), -self.input))
        # print("sigmoid backward", self.input.shape, grad_output.shape, gradient.mul(grad_output).shape)
        return torch.mul(gradient,grad_output)

    def zero_grad(self):
        pass


class SoftMax():
    def __init__(self, context_len):
        self.is_trainable = False
        self.context_len = context_len

    def forward(self, input):
        # print("softmax forward",input.shape)
        input = input.reshape(input.size(0), self.context_len, -1)
        self.input = input

        input -= input.max(2)[0].unsqueeze(2).repeat_interleave(input.size(2), dim=2)
        temp1 = torch.exp(input)
        temp2 = (1 / temp1.sum(2)).unsqueeze(dim=2)
        self.prob = torch.mul(temp1, temp2.repeat_interleave(temp1.size(2), dim=2))

        self.gradient = self.prob.clone()
        return self.prob

    def backward(self, grad_output):
        # print("softmax backward", self.input.shape, grad_output.shape, torch.add(self.gradient, -grad_output).shape)
        return (self.gradient -grad_output)

    def zero_grad(self):
        pass


class CrossEntropy():
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.is_trainable = False

    def get_indicator_matrix(self, targets, mask_zero_index=True):
        batch_size, context_len = targets.shape
        indicator_matrix = torch.zeros((batch_size, context_len, self.nclasses))
        for i in range(batch_size):
            index = (targets[i] > 0).nonzero()
            indicator_matrix[i, index, targets[i, index]] = 1.0
        return indicator_matrix

    def forward(self, prob, y):
        self.y = y
        self.indicator_matrix = self.get_indicator_matrix(y)
        output  = -(prob[self.indicator_matrix==1] + TINY).log().sum()
        return output

    def backward(self, grad_output=1):
        # print("cross entropy", self.y.shape, grad_output, self.get_indicator_matrix(self.y).shape)
        return self.get_indicator_matrix(self.y)

    def zero_grad(self):
        pass