## import libraries for training
import warnings
import torch.optim
from sklearn.metrics import classification_report, cohen_kappa_score
from src.utils import *
warnings.filterwarnings('ignore')

class Validator:
    # Validating the model
    def evaluate(self, val_loader,model):
        model.cuda()
        model.eval()
        model.training=False
        map = AverageMeter()
        
        all_gts = []
        all_preds = []
        with torch.no_grad():
            for i, (images,target,fnames) in enumerate(val_loader):
                img = images.cuda(non_blocking=True)
                label = target.cuda(non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    logits = model(img)
                    preds = logits.softmax(1)
                
                all_gts.extend(label.cpu())
                all_preds.extend(preds.cpu().argmax(axis=1))
                
                valid_map5, valid_acc1, valid_acc5 = self.map_accuracy(preds, label)
                map.update(valid_map5,img.size(0))
        print(classification_report(all_gts, all_preds))
        print("Cohen's Kappa: ",cohen_kappa_score(all_gts, all_preds))
        return map.avg

    ## Computing the mean average precision, accuracy 
    def map_accuracy(self, probs, truth, k=5):
        with torch.no_grad():
            value, top = probs.topk(k, dim=1, largest=True, sorted=True)
            correct = top.eq(truth.view(-1, 1).expand_as(top))

            # top accuracy
            correct = correct.float().sum(0, keepdim=False)
            correct = correct / len(truth)

            accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
            map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
            acc1 = accs[0]
            acc5 = accs[1]
            return map5, acc1, acc5
        
