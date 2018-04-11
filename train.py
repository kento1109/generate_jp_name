from __future__ import unicode_literals, print_function, division
import random
import csv
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import cPickle
from model import LSTM


def inputTensor(input_idx):
    tensor = torch.LongTensor(input_idx)
    return autograd.Variable(tensor)


def targetTensor(input_idx, char2idx):
    input_idx = input_idx[1:]
    input_idx.append(char2idx['EOS'])
    tensor = torch.LongTensor(input_idx)
    return autograd.Variable(tensor)


def train(model, criterion, input, target):
    hidden = model.initHidden()

    model.zero_grad()

    output, _ = model(input, hidden)
    _, predY = torch.max(output.data, 1)
    loss = criterion(output, target)

    loss.backward()

    return loss.data[0] / input.size()[0]


def read_csv(filname):
    names_str = []
    with open(filname) as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            names_str.append(row[4].decode('utf-8'))
    return names_str


def main():
    names_str = read_csv(filname='data/names/names.csv')
    all_char_str = set([char for name in names_str for char in name])
    char2idx = {char: i for i, char in enumerate(all_char_str)}
    char2idx['EOS'] = len(char2idx)
    # save char dictionary
    cPickle.dump(char2idx, open("dic.p", "wb"))

    names_idx = [[char2idx[char_str] for char_str in name_str]
                 for name_str in names_str]

    # build model
    model = LSTM(input_dim=len(char2idx), embed_dim=100, hidden_dim=128)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    n_iters = 5

    for iter in range(1, n_iters + 1):

        # data shuffle
        random.shuffle(names_idx)

        total_loss = 0

        for i, name_idx in enumerate(names_idx):
            input = inputTensor(name_idx)
            target = targetTensor(name_idx, char2idx)

            loss = train(model, criterion, input, target)
            total_loss += loss

            optimizer.step()

        print(iter, "/", n_iters)
        print("loss {:.4}".format(float(total_loss / len(names_idx))))

        # save trained model
        torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    main()
