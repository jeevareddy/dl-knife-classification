## import libraries for training
import warnings
from datetime import datetime
from timeit import default_timer as timer
import torch
from torch import optim
from torch.optim import lr_scheduler
from src.utils import *

warnings.filterwarnings("ignore")


class Trainer:
    log = Logger()

    def get_dynamic_file_name(self, epoch=config.epochs):
        return f"{config.model_name}{'' if config.class_weights is None else '_Cw'}{'_Aug' if len(config.dataAugmentations) != 0  else ''}{'_Fl' if config.freezeLayer else ''}_E{epoch}_B{config.batch_size}_LR{config.learning_rate}_WD{config.weight_decay}_{config.optim}_{config.criterion}"

    def initLogger(self):
        ## Writing the loss and results
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")
        filename = f"logs/train_{self.get_dynamic_file_name(epoch=config.epochs)}@1.txt"
        if os.path.exists(filename):
            filename, itr = filename.split("@")
            filename = filename + f"@{int(itr.split('.')[0])+1}.txt"
        self.log.open(filename)
        self.log.write(f"\nConfig:\n")
        self.log.write(config.toString())
        self.log.write(
            "\n----------------------------------------------- [START %s] %s\n\n"
            % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-" * 51)
        )
        self.log.write(
            "                           |----- Train -----|----- Valid----|---------|\n"
        )
        self.log.write(
            "mode     iter     epoch    |       loss      |        mAP    | time    |\n"
        )
        self.log.write(
            "-------------------------------------------------------------------------------------------\n"
        )

    ## Training the model
    def train_epoch(
        self, train_loader, model, criterion, optimizer, epoch, valid_accuracy, start
    ):
        losses = AverageMeter()
        model.train()
        model.training = True
        for i, (images, target, fnames) in enumerate(train_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
            loss = criterion(logits, label)
            losses.update(loss.item(), images.size(0))
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            self.scheduler.step()

            print("\r", end="", flush=True)
            message = "%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s" % (
                "train",
                i,
                epoch,
                losses.avg,
                valid_accuracy[0],
                time_to_str((timer() - start), "min"),
            )
            print(message, end="", flush=True)
        self.log.write("\n")
        self.log.write(message)

        return [losses.avg]

    # Validating the model
    def evaluate(self, val_loader, model, criterion, epoch, train_loss, start):
        model.cuda()
        model.eval()
        model.training = False
        map = AverageMeter()
        with torch.no_grad():
            for i, (images, target, fnames) in enumerate(val_loader):
                img = images.cuda(non_blocking=True)
                label = target.cuda(non_blocking=True)

                with torch.cuda.amp.autocast():
                    logits = model(img)
                    preds = logits.softmax(1)

                valid_map5, valid_acc1, valid_acc5 = self.map_accuracy(preds, label)
                map.update(valid_map5, img.size(0))
                print("\r", end="", flush=True)
                message = (
                    "%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s"
                    % (
                        "val",
                        i,
                        epoch,
                        train_loss[0],
                        map.avg,
                        time_to_str((timer() - start), "min"),
                    )
                )
                print(message, end="", flush=True)
            self.log.write("\n")
            self.log.write(message)
        return [map.avg]

    ## Computing the mean average precision, accuracy
    def map_accuracy(self, probs, truth, k=5):
        with torch.no_grad():
            value, top = probs.topk(k, dim=1, largest=True, sorted=True)
            correct = top.eq(truth.view(-1, 1).expand_as(top))

            # top accuracy
            correct = correct.float().sum(0, keepdim=False)
            correct = correct / len(truth)

            accs = [
                correct[0],
                correct[0] + correct[1] + correct[2] + correct[3] + correct[4],
            ]
            map5 = (
                correct[0] / 1
                + correct[1] / 2
                + correct[2] / 3
                + correct[3] / 4
                + correct[4] / 5
            )
            acc1 = accs[0]
            acc5 = accs[1]
            return map5, acc1, acc5

    def __init__(self, model, train_loader, val_loader, *args, **kwargs):
        ############################# Parameters #################################
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.initLogger()
        self.initOptimizer()
        self.initCriterion()

        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=config.epochs * len(train_loader),
            eta_min=0,
            last_epoch=-1,
        )

    def initOptimizer(self):
        if config.optim == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        if config.optim == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        if config.optim == "radam":
            self.optimizer = optim.RAdam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        if config.optim == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        if config.optim == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        if config.optim == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        if config.optim == "adamax":
            self.optimizer = optim.Adamax(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        if config.optim == "LBFGS":
            self.optimizer = optim.LBFGS(
                self.model.parameters(), lr=config.learning_rate
            )

    def initCriterion(self):
        if config.criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss(weight= None if config.class_weights is None else  torch.HalfTensor(config.class_weights).cuda() ).cuda()
        if config.criterion == "mae":
            self.criterion = nn.L1Loss().cuda()

    def train(self):
        ############################# Training #################################
        val_metrics = [0]
        start = timer()
        # train
        for epoch in range(0, config.epochs):
            lr = get_learning_rate(self.optimizer)
            train_metrics = self.train_epoch(
                self.train_loader,
                self.model,
                self.criterion,
                self.optimizer,
                epoch,
                val_metrics,
                start,
            )
            val_metrics = self.evaluate(
                self.val_loader, self.model, self.criterion, epoch, train_metrics, start
            )
            ## Saving the model
            if not os.path.exists("./checkpoints/"):
                os.mkdir("./checkpoints/")
            if not os.path.exists(f"./checkpoints/{config.model_name}/"):
                os.mkdir(f"./checkpoints/{config.model_name}/")
            filename = f"checkpoints/{config.model_name}/{self.get_dynamic_file_name(epoch=str(epoch + 1))}@1.pt"
            if os.path.exists(filename):
                filename, itr = filename.split("@")
                filename = filename + f"@{int(itr.split('.')[0])+1}.pt"
            torch.save(self.model.state_dict(), filename)
