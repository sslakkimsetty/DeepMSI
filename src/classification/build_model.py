"""
This file is for creating a classification model according to 
specifications
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from classification.cnn import CNN
from classification.rnn import RNN
import logging

import logging


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('my_logger')


def build_base_model(model_opt, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the classification model.
    """
    assert model_opt.model_type in ["cnn", "rnn"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Build CNN model.
    if model_opt.model_type == "cnn":
      #model = CNN(model_opt.num_layers, model_opt.input_size, 
      #            model_opt.kernel_width, model_opt.chanels, model_opt.dropout)
      model = CNN()
    
    # Build RNN model  
    elif model_opt.model_type == "rnn":
      model = CNN(model_opt.num_layers, model_opt.input_size, 
                  model_opt.kernel_width, model_opt.chanels, model_opt.dropout)



    # use cuda if gpu is available
    device = torch.device("cuda" if gpu else "cpu")




    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)



    # model to device

    model.to(device)

    return model


def build_model(model_opt, gpu, checkpoint):
    """ Build the Model """
    logger.info('Building model...')
    
    model = build_base_model(model_opt,
                             gpu, checkpoint)
    logger.info(model)
    return model