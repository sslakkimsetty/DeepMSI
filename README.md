# DeepMSI
This package implements deep learning methods for MSI classification and clustering. The classification methods include supervised training and semi-supervised training where only tissue-level labels are available.

## Tissue classification

### Load the package 

```python
from classification import *
```

### Config the classification model

```python
model_opts = model_opt( model_type = 'cnn', gpu = False, param_init = 0.0, 
                    num_layers = 3, kernel_width = [16, 22, 38],
                    chanels = [8,16,3]
)
```
`model_type`  specifies the backbone of the classifier. It has the options of 'cnn' and 'rnn'.

`gpu` specifies whether to use GPU 

`param_init` specifies the initialization of parameters, '0.0' for default initialization in pytorch, and other options can be 'uniform', 'xavier_uniform', 'xavier_normal'.

`num_layers` specifies the number of convolutional or recurrent layers.


### Build a classifier

```python
classifier = classify(model_opts, data = PATH_TO_DATA, batch_size = 32, optim = 'Adam', lr = 0.001, num_epochs = 10)
```
`model_opts`: the classification model configurations

`data`: path to the data file. The data file should be in .csv format

`bacth_size`: sample size for a mini-batch

`optim`: optimizer

`lr`: learning rate

`num_epochs`: number of training epochs


### Train the classifier

```python
classifier.train()
```

### Visualizations

```python
viz(data = PATH_TO_DATA, classification = classifier).plot_data()
```
visualize the groud truth labels of data

```python
viz(data = PATH_TO_DATA, classification = classifier).plot_results()
```
visualize the predicted labels of the classifier
