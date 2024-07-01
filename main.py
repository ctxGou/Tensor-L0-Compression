import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from model.lenet5 import *
import data
import utils

import datetime
import tensorly as tl

tl.set_backend('pytorch')

log_file = './log//lenet5.txt'
result_csv = './log//lenet5.csv'

model_names = [ 
    ('lenet5_tr_25_stg', 0.2),
    ('lenet5_tr_25_stg', 1),
    
               ]

reference_model = get_lenet5_model('lenet5')
model_func = get_lenet5_model

lr = 5e-4
n_epochs = 150
batch_size = 128
optim = "Adam"


target_cr = 12312321

criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def main(model_name, lambd, sigma):
    criterion = nn.CrossEntropyLoss()
    model = model_func(model_name, sigma=sigma).to(device)

    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)

    train_loader, valid_loader, test_loader = data.get_mnist_loaders(
        batch_size=batch_size, ram=True)

    train_batch_time = datetime.timedelta()

    for n in range(n_epochs):

        losses, reg, batch_time = train_epoch(
            train_loader, model, criterion, optimizer, n, device, lambd
        )
        train_batch_time += batch_time
        scheduler.step()

        val_top1, val_top5, val_losses, _ = validate(
            valid_loader, model, criterion, n)

        msg = f'[{n}/{n_epochs}] Loss:{losses}, Reg:{reg}, CR:{reference_model.count()/model.count()} Time:{batch_time}, Structure: {model.structure()}\n'
        msg += f'   [VAL] Loss:{val_losses}, acc@1:{val_top1}, acc@5:{val_top5}\n'
        utils.printtt(msg, log_file)

    train_batch_time = train_batch_time / n_epochs

    top1, top5, losses, inference_time = validate(
        test_loader, model, criterion, n_epochs)

    msg = f'\n[{model_name} TEST] Loss:{losses}, acc@1:{top1}, acc@5:{top5}, Time:{inference_time}\n'

    reference_storage = reference_model.count()
    new_storage = model.count()

    msg += f'Compression: {reference_storage/new_storage:.3f}, Structure: {model.structure()}\n\n\n'

    utils.printtt(msg, log_file)

    return top1.average(), top5.average(), reference_storage/new_storage, train_batch_time, inference_time


def train_epoch(train_loader, model, criterion, optimizer, epoch, device, lambd):


    model.train()
    start = datetime.datetime.now()

    training_losses = Registrar()
    training_reg = Registrar()

    for i, (features, target) in enumerate(train_loader):
        features = features.to(device)
        target = target.to(device)
        output = model(features)
        loss = criterion(output, target)
        training_losses.update(loss.item())

        reg = model.regularizer()
        training_reg.update(reg.item())

        cr = reference_model.count()/model.count()
        if cr < target_cr:
            loss += lambd*reg
        else:
            fix_gate(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    duration = datetime.datetime.now() - start
    avg_batch_time = duration/(i+1)

    return training_losses, training_reg, avg_batch_time


def validate(loader, model, criterion, epoch):
    model.eval()

    inference_time = datetime.datetime.now()
    losses = Registrar()
    top1, top5 = Registrar(), Registrar()

    with torch.no_grad():
        for i, (features, target) in enumerate(loader):
            features = features.to(device)
            target = target.to(device)
            output = model(features)
            loss = criterion(output, target)
            losses.update(loss.item())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item())
            top5.update(acc5.item())

    inference_time = datetime.datetime.now() - inference_time

    return top1, top5, losses, inference_time


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Registrar(object):
    def __init__(self):
        self.training_losses = []

    def update(self, loss):
        self.training_losses.append(loss)

    def average(self):
        return sum(self.training_losses)/len(self.training_losses)

    def __str__(self):
        return f"{self.average():.5f}"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    import pandas as pd

    df = pd.DataFrame(columns=["Model", "acc@1", "acc@5", "cr", "train_time", "inference_time",
                               "lambda", "sigma", "lr", "n_epochs", "batch_size", "optimizer"])
    df.to_csv(result_csv, encoding='utf-8', index=False, mode="a")

    runs = 3

    for m, lambd in model_names:

        for i in range(runs):

            set_seed(i)

            sigma = 0.5
            top1, top5, cr, train_time, test_time = main(m, lambd, sigma)
            new_row = [
                f'{m}_{i}',
                top1, top5, cr, train_time, test_time,
                lambd, sigma, lr,
                n_epochs, batch_size, optim
            ]

            new_row = pd.DataFrame(new_row).T
            new_row.to_csv(result_csv, encoding='utf-8',
                        index=False, header=False, mode="a")
