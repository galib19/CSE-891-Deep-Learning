# main.py

import math
import data
import optim
from model import Model

# parameters
args = {'batch_size': 100,                    # size of a mini-batch
        'learning_rate': 1e-5,                   # learning rate
        'momentum': 0.9,                      # decay parameter for the momentum vector
        'weight_decay': 0,                    # L2 regularization on the weights
        'epochs': 50,                         # maximum number of epochs to run
        'context_len': 4,                     # number of context words used
        'embedding_dim': 16,                  # number of dimensions in embedding
        'vocab_size': 251,                    # number of words in vocabulary
        'num_hid': 128,                       # number of hidden units
        'model_file': 'model.pk',             # filename to save best model
}

# dataloaders
loader_test = data.DataLoader(args['batch_size'], 'Test')
loader_valid = data.DataLoader(args['batch_size'], 'Valid')
loader_train = data.DataLoader(args['batch_size'], 'Train')

# create model
model = Model(args, loader_train.vocab)

# initialize optimizer
optimizer = optim.SGDMomentum(model.model, learning_rate=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay'])

best_validation_loss = 10000000.0

print("")
for epoch in range(args['epochs']):
    # training
    total_loss = 0
    for batch in range(math.ceil(loader_train.get_size() / args['batch_size'])):
        model.model.zero_grad()
        input = loader_train.get_batch()
        batch_size = input.size(0)
        mask = loader_train.sample_mask(batch_size)
        input_masked = input * (1 - mask)
        target_masked = input * mask
        output = model.model.forward(input_masked)
        loss = model.criterion.forward(output, target_masked)
        grad_loss = model.criterion.backward()
        model.model.backward(grad_loss)
        model.model = optimizer.step(model.model)
        total_loss += loss.item()
    total_loss = total_loss / loader_train.get_size()
    print("Training Epoch: [%d]\t Loss:  %.4f \t" % (epoch, total_loss))

    # validation
    total_loss = 0
    for batch in range(math.ceil(loader_valid.get_size() / args['batch_size'])):
        model.model.zero_grad()
        input = loader_train.get_batch()
        batch_size = input.size(0)
        mask = loader_valid.sample_mask(batch_size)
        input_masked = input * (1 - mask)
        target_masked = input * mask
        output = model.model.forward(input_masked)
        loss = model.criterion.forward(output, target_masked)
        total_loss += loss.item()
    total_loss = total_loss / loader_valid.get_size()
    print("Validation Epoch: [%d]\t Loss:  %.4f \t" % (epoch, total_loss))
    if best_validation_loss > total_loss:
        best_validation_loss = total_loss
        model.save(args['model_file'])
    else:
        print("Validation loss increased, finishing training.")
        break
    print('')

train_CE = model.evaluate(loader_train, args['batch_size'])
print('Final training cross-entropy: {:1.3f}'.format(train_CE))
valid_CE = model.evaluate(loader_valid, args['batch_size'])
print('Final validation cross-entropy: {:1.3f}'.format(valid_CE))
test_CE = model.evaluate(loader_test, args['batch_size'])
print('Final test cross-entropy: {:1.3f}'.format(test_CE))

model.tsne_plot()