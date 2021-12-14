class model_opt():



    def __init__(self, model_type = 'cnn', param_init = 0.0, param_init_glorot = None):
        super(model_opt, self).__init__()


        

        self.model_type = model_type
        self.param_init = param_init
        self.param_init_glorot = param_init_glorot