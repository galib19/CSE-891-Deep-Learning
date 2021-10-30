# optim.py

import torch


class SGD:
    def __init__(self, learning_rate, weight_decay=0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def step(self, model):
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                grad_bias = torch.add(layer.grad_bias, torch.mul(layer.bias, self.weight_decay))
                grad_weight = torch.add(layer.grad_weight, torch.mul(layer.weight, self.weight_decay))
                delta_bias = torch.mul(grad_bias, self.lr)
                delta_weight = torch.mul(grad_weight, self.lr)
                layer.bias = torch.add(layer.bias, -delta_bias)
                layer.weight = torch.add(layer.weight, -delta_weight)
        return model

    def zero_grad(self):
        pass


class SGDMomentum:
    def __init__(self, model, learning_rate, momentum, weight_decay=0):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.cache = []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                self.cache.append(dict(grad_weight=torch.zeros(layer.weight.size()), grad_bias=torch.zeros(layer.bias.size())))
            else:
                self.cache.append(dict(grad_weight=torch.zeros(1), grad_bias=torch.zeros(1)))

    def step(self, model):
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                # print(i, model.layers[i])
                grad_bias = torch.add(layer.grad_bias, torch.mul(layer.bias, self.weight_decay))
                grad_weight = torch.add(layer.grad_weight, torch.mul(layer.weight, self.weight_decay))

                grad_bias = torch.add(torch.mul(self.cache[i]['grad_bias'], self.momentum), grad_bias)
                grad_weight = torch.add(torch.mul(self.cache[i]['grad_weight'], self.momentum), grad_weight)

                self.cache[i]['grad_bias'] = grad_bias
                self.cache[i]['grad_weight'] = grad_weight
                layer.bias -=  torch.mul(layer.grad_bias, self.lr)
                layer.weight -= torch.mul(layer.grad_weight, self.lr)
        return model

    def zero_grad(self):
        pass