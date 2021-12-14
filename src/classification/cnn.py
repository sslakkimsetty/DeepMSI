import torch
import torch.nn as nn

class CNN(nn.Module):
    """ 
    Stacked CNN class
    Args:
      input_size (int): number of mass spectra for a pixel
      num_layers (int): number of stacked convolutional layers
      kernel_width (list): kernel width for each layer
      chanels (list): number of output chanels for each layer

    """


    def __init__(self, num_layers = 3, input_size = 1, kernel_width = [16, 22, 38],
                 chanels = [8,16,3], dropout=0.05):
        super(CNN, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.kernel_width = kernel_width
        self.chanels = chanels
        self.dropout = dropout

        assert (len(self.kernel_width) == self.num_layers),\
          """ The length of kernel width should equal to the number of CNN layers """
        assert (len(self.chanels) == self.num_layers),\
          """ The length of chanels should equal to the number of CNN layers """
        
        self.layers = nn.ModuleList()
        in_chanels = input_size
        for i in range(num_layers):

          out_chanels = self.chanels[i]
          self.layers.append(
              nn.Sequential(nn.Conv1d(in_chanels, out_chanels, kernel_size=self.kernel_width[i], stride=1, bias=False),
                              nn.BatchNorm1d(out_chanels),
                              nn.ReLU(),
                              nn.MaxPool1d(2),
                              nn.BatchNorm1d(out_chanels),
                              nn.ReLU()))
          in_chanels = out_chanels
          
        self.fc = nn.Sequential(nn.Linear(48*3, 2), nn.Softmax())
        

    def forward(self, x):
        for conv in self.layers:
            x = conv(x)
        x = x.view(-1, 48*3)
        x = self.fc(x)

        return x