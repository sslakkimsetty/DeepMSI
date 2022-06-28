import torch
import torch.nn as nn

class RNN(nn.Module):
    """ 
    Stacked RNN class
    Args:
      input_size (int): number of mass features in a spectra
      num_layers (int): number of stacked RNN layers
      hidden_dim (int): dimension of the hidden layer

    """


    def __init__(self, num_layers = 3, input_size = 596, hidden_dim = 256, dropout=0.05):
        super(RNN, self).__init__()


        

        self.hidden_dim = hidden_dim
        

        self.num_layers = num_layers
        
        # RNN
        self.rnn = nn.RNN(input_size, hidden_dim, num_layers, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out