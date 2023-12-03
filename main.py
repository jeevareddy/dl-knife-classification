import argparse
import timm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.train import Trainer
from src.test import Validator
from src.data import knifeDataset
import numpy as np
from src.config import *
import random
random.seed(40)

def getModel(modelName="tf_efficientnet_b0", checkpoint_path=None, classes=192):
    model = timm.create_model(
        modelName, pretrained=(checkpoint_path is None), num_classes=classes
    )
    if checkpoint_path:
        print("loading trained model")
        model.load_state_dict(torch.load(checkpoint_path))
    if config.freezeLayer:
        print("Freezing hidden layers")
        for param in model.parameters():
            param.requires_grad = False
        model.reset_classifier(classes)
    return model


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Run train script", action="store_true")
    parser.add_argument("--eval", help="Evaluate provided model", action="store_true")
    parser.add_argument("--model", default="tf_efficientnet_b0", help="Model name")
    parser.add_argument("--freeze", help="Freeze hidden layers", action="store_true")
    parser.add_argument("--checkpoint", help="Path to Saved Model Weight")
    parser.add_argument(
        "--criterion",
        default="crossentropy",
        help="Criterion / Loss Function",
        choices=["crossentropy", "mae"],
    )
    parser.add_argument(
        "--class-weights",
        help="Incorporate class weights to criterion",
        action="store_true",
    )
    parser.add_argument(
        "--optim",
        default="adam",
        help="Optimizer",
        choices=["adam", "adamw", "radam", "adadelta", "adamax", "sgd", "rmsprop"],
    )
    parser.add_argument(
        "--augs",
        nargs="+",
        help="Data Augmentations",
        choices=["color-jitter", "rotate", "sharpness", "v-flip", "h-flip"],
    )

    ## Configs
    parser.add_argument(
        "--num_classes", default=192, type=int, help="Number of Classes"
    )
    parser.add_argument("--img_width", default=224, type=int, help="Image width")
    parser.add_argument("--img_height", default=224, type=int, help="Image height")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch Size")
    parser.add_argument("--data_dir", help="Dataset Directory")
    parser.add_argument("--epochs", default=20, type=int, help="Training Epochs")
    parser.add_argument("-lr", default=0.00005, type=float, help="Learning Rate")
    parser.add_argument("-wd", default=0.00005, type=float, help="Weight Decay")

    args = parser.parse_args()
    config.parseArgs(parsed_args=args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################## load file and get splits #############################
    train_imlist = pd.read_csv("dataset/train.csv")
    
    # Calculate Class Weights
    if (args.class_weights):
        class_hist = train_imlist['Label'].value_counts(sort=False).to_numpy(np.float16)
        config.class_weights = class_hist/len(class_hist)
    
    train_gen = knifeDataset(train_imlist, mode="train", data_dir=args.data_dir)
    train_loader = DataLoader(
        train_gen,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        collate_fn=collate_fn,
    )
    val_imlist = pd.read_csv("dataset/test.csv")
    val_gen = knifeDataset(val_imlist, mode="val", data_dir=args.data_dir)
    val_loader = DataLoader(
        val_gen,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=collate_fn,
    )

    model = getModel(modelName=args.model, checkpoint_path=args.checkpoint).to(device)

    if args.train:
        print("Initiating training")
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader)
        trainer.train()
    elif args.eval:
        print("Evaluating trained model")
        validator = Validator()
        map = validator.evaluate(model=model, val_loader=val_loader)
        print("mAP =", map)


### Data Augmentations
# Flip, Rotate, Blur, Scale, Color Jitter
