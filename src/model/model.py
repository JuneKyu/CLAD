import torch.nn as nn
from torch.autograd import Variable 
import torch
import pdb

class Encoder(nn.Module):
    
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        
        self.drop = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.hidden_size, args.input_size)

        if args.cell_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.cell_type)(input_size= args.input_size,\
                                                   hidden_size = args.hidden_size,\
                                                   num_layers = args.nlayers, \
                                                   dropout=args.dropout,  batch_first = True)
    
    def init_weights(self):
        initrange = 0.1
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def forward(self, input, hidden, return_hiddens=False, noise=False):
        output, hidden = self.rnn(input, hidden)     
        output = self.linear(output.contiguous().view(-1,self.args.hidden_size))
        output = output.contiguous().view(input.size()[0], -1, self.args.input_size)
        return output, hidden
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data ############# 이게 무엇
        if self.args.cell_type == 'LSTM':
            return (Variable(weight.new(self.args.nlayers, bsz, self.args.hidden_size).zero_()),
                    Variable(weight.new(self.args.nlayers, bsz, self.args.hidden_size).zero_()))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return Variable(h.data)

    def extract_hidden(self, hidden):
        if self.args.cell_type == 'LSTM':
            return hidden[0][-1].data.cpu()  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer

        
        
class Decoder(nn.Module):

    def __init__(self,args):
        super(Decoder, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.hidden_size, args.input_size)

        if args.cell_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.cell_type)(input_size= args.input_size,\
                                                   hidden_size = args.hidden_size,\
                                                   num_layers = args.nlayers, \
                                                   dropout=args.dropout,  batch_first = True)


    def init_weights(self):
        initrange = 0.1
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)
        
    def forward(self, input, hidden, return_hiddens=False, noise=False):
        output, hidden = self.rnn(input, hidden)     
        output = self.linear(output.contiguous().view(-1,self.args.hidden_size))
        output = output.contiguous().view(input.size()[0], -1, self.args.input_size)

        return output, hidden




    def init_hidden(self, bsz):
        weight = next(self.parameters()).data ############# 이게 무엇
        if self.args.cell_type == 'LSTM':
            return (Variable(weight.new(self.args.nlayers, bsz, self.args.hidden_size).zero_()),
                    Variable(weight.new(self.args.nlayers, bsz, self.args.hidden_size).zero_()))

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return Variable(h.data)

    def extract_hidden(self, hidden):
        if self.args.cell_type == 'LSTM':
            return hidden[0][-1].data.cpu()  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer
        
        