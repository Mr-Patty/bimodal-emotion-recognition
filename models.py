import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import math
import librosa
import librosa.display
import scipy, matplotlib.pyplot as plt, IPython.display as ipd

from scipy.io import wavfile

import librosa.display
import soundfile as sf

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class AttentionRNNModel(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embedding_length, bidirectional=False, num_layers=1):
        super(AttentionRNNModel, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.bidirectional = bidirectional
        self.num_layers = num_layers

#         self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
#         self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional=self.bidirectional, num_layers=self.num_layers)
        self.label = nn.Linear(hidden_size, output_size)
        #self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------
    
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                hidden.size() = (batch_size, hidden_size)
                attn_weights.size() = (batch_size, num_seq)
                soft_attn_weights.size() = (batch_size, num_seq)
                new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input, batch_size=None):

        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
    
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """

#         input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2).float()
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).float())
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).float())
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).float())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).float())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits
    
class AttentionCNNModel(torch.nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, kernel_size, embedding_length=128, stride=1, padding=0):
        super(AttentionCNNModel, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv1 = nn.Conv2d(embedding_length, 64, 3)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.pool3 = nn.MaxPool2d(3)
        