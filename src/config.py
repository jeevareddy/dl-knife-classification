from torchvision import transforms as T


class DefaultConfigs:
    n_classes = 192  ## number of classes
    img_weight = 224  ## image width
    img_height = 224  ## image height
    batch_size = 16  ## batch size
    epochs = 20  ## epochs
    learning_rate = 0.00005  ## learning rate
    weight_decay = 0.00005  ## learning rate
    model_name = ""
    freezeLayer = False
    criterion = "crossentropy"
    optim = "adam"
    dataAugmentations = []
    class_weights = None

    _augmentations = {
        "color-jitter": T.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
        "rotate": T.RandomRotation(degrees=(0, 90)),
        "sharpness": T.RandomAdjustSharpness(0.8, p=0.5),
        "v-flip": T.RandomVerticalFlip(p=0.5),
        "h-flip": T.RandomHorizontalFlip(p=0.5)
    }

    def parseArgs(self, parsed_args=None):
        if parsed_args:
            self.n_classes = parsed_args.num_classes
            self.img_weight = parsed_args.img_width
            self.img_height = parsed_args.img_height
            self.batch_size = parsed_args.batch_size
            self.epochs = parsed_args.epochs
            self.learning_rate = parsed_args.lr
            self.weight_decay = parsed_args.wd
            self.model_name = parsed_args.model
            self.freezeLayer = parsed_args.freeze
            self.criterion = parsed_args.criterion
            self.optim = parsed_args.optim
            if parsed_args.augs:
                self.dataAugmentations = [
                    self._augmentations[aug] for aug in parsed_args.augs
                ]

    def toString(self):
        return f"""n_classes: {self.n_classes}
img_weight: {self.img_weight}
img_height: {self.img_height}
batch_size: {self.batch_size}
epochs: {self.epochs}
learning_rate: {self.learning_rate}
weight_decay: {self.weight_decay}
model_name: {self.model_name}
freezeLayer: {self.freezeLayer}
criterion: {self.criterion}
optim: {self.optim}
dataAugmentations: {self.dataAugmentations}"""


config = DefaultConfigs()
