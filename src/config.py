class DefaultConfigs(object):
    n_classes = 192  ## number of classes
    img_weight = 224  ## image width
    img_height = 224  ## image height
    batch_size = 16 ## batch size
    epochs = 20    ## epochs
    learning_rate=0.00005  ## learning rate
    model_name = ''
    
    
    def parseArgs(self, parsed_args=None):
        if parsed_args:
            self.n_classes = parsed_args.num_classes
            self.img_weight = parsed_args.img_width
            self.img_height = parsed_args.img_height
            self.batch_size = parsed_args.batch_size
            self.epochs = parsed_args.epochs
            self.learning_rate = parsed_args.lr
            self.model_name = parsed_args.model
        
config = DefaultConfigs()
