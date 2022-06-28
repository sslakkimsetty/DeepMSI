import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from classification.classification import classification

class viz(object):
    """ 
    visualization of data and results
    args:
      data:
      classification (obj) 

    """


    def __init__(self, data, classification = None):
        super(viz, self).__init__()

        self.data = data
        self.classification = classification

    def plot_data(self):
      data = pd.read_csv(self.data)
      x = data['x']
      y = data['y']
      z = data['label']
      graph = plt.scatter(x, y, c=z, s=3)

      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('True Labels')
      plt.colorbar(graph)


    def plot_results(self):
      assert (self.classification),\
          """ No results object to be plot! """
      data = pd.read_csv(self.data)
      x = data['x']
      y = data['y']
      pred_label, pred_prob = self.classification.val()
      graph = plt.scatter(x, y, c=pred_label, s=3)

      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('Predicted Labels')
      plt.colorbar(graph)