#coding: utf-8
import cPickle
from model import LSTM
from train import inputTensor
import torch


def sample(start_letter='ア', max_length=5):
    sample_char_idx = [char2idx[start_letter]]

    input = inputTensor(sample_char_idx)

    hidden = model.initHidden()

    output_name = start_letter

    for i in range(max_length):
        output, hidden = model(input, hidden)
        _, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == char2idx['EOS']:
            break
        else:
            letter = idx2char[topi]
            output_name += letter
        input = inputTensor([topi])

    return output_name


def samples(start_letters='アイウ'):
    for start_letter in start_letters:
        print(sample(start_letter))


def main():
    samples(u'アスナ')


if __name__ == '__main__':
    # load dic
    char2idx = cPickle.load(open("dic.p", "rb"))
    idx2char = {v: k for k, v in char2idx.items()}
    # build model
    model = LSTM(input_dim=len(char2idx), embed_dim=100, hidden_dim=128)
    model.load_state_dict(torch.load('model.pt'))
    main()
