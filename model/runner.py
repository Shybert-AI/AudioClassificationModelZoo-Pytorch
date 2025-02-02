import torch
from tqdm import tqdm
import torch.nn.functional as F

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)


def train_epoch(
    model=None,
    optimizer=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
    loss_meter=None,
    score_meter=None,
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    avg_f1_meter = AverageMeter()
    avg_pre_meter = AverageMeter()
    avg_rec_meter = AverageMeter()
    logs = {}

    model.to(device).train()
    with tqdm(dataloader, desc="Train") as iterator:

        for sample in iterator:
            x = sample[0].to(device)
            y = sample[1].to(device)
            n = x.shape[0]
            optimizer.zero_grad()
            if model.name =="DTFAT":
                x = x.repeat(1, 1, 4,1)
                x = F.pad(x, (0, 28, 0, 0, 0, 0))
            elif model.name =="ASTModel" or model.name == "CAMPPlus":
                x = F.pad(x, (0, 28, 0, 0, 0, 0))
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            pre_metric= metric(outputs, y)
            score_meter.update(pre_metric[0], n=n)
            avg_f1_meter.update(pre_metric[1], n=n)
            avg_pre_meter.update(pre_metric[2], n=n)
            avg_rec_meter.update(pre_metric[3], n=n)

            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name[0]: score_meter.avg})
            logs.update({metric.name[1]: avg_pre_meter.avg})
            logs.update({metric.name[2]: avg_rec_meter.avg})
            logs.update({metric.name[3]: avg_f1_meter.avg})

            iterator.set_postfix_str(format_logs(logs))
    return logs


def valid_epoch(
    model=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    avg_f1_meter = AverageMeter()
    avg_pre_meter = AverageMeter()
    avg_rec_meter = AverageMeter()
    logs = {}
    model.to(device).eval()
    correct_predictions = 0
    total_pixels = 0
    with tqdm(dataloader, desc="Valid") as iterator:
        for sample in iterator:
            x = sample[0].to(device)
            y = sample[1].to(device)
            n = x.shape[1]

            with torch.no_grad():
                if model.name == "DTFAT":
                    x = x.repeat(1, 1, 4, 1)
                    x = F.pad(x, (0, 28, 0, 0, 0, 0))
                elif model.name == "ASTModel" or model.name == "CAMPPlus":
                    x = F.pad(x, (0, 28, 0, 0, 0, 0))
                outputs = model.forward(x)
                loss = criterion(outputs, y)

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            pre_metric = metric(outputs, y)
            score_meter.update(pre_metric[0], n=n)
            avg_f1_meter.update(pre_metric[1], n=n)
            avg_pre_meter.update(pre_metric[2], n=n)
            avg_rec_meter.update(pre_metric[3], n=n)
            
            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name[0]: score_meter.avg})
            logs.update({metric.name[1]: avg_pre_meter.avg})
            logs.update({metric.name[2]: avg_rec_meter.avg})
            logs.update({metric.name[3]: avg_f1_meter.avg})
            iterator.set_postfix_str(format_logs(logs))

    return logs

