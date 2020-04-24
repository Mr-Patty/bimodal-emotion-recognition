import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def train_cycle(model, optimizer, loss_func, n_epoch, train_loader, validation_loader, device):
    model.train()
    epoch_losses = []
    epoch_losses_val = []
    for epoch in range(n_epoch):
        print("Epoch: {}".format(epoch))
        losses = []
        for batch in tqdm(train_loader):
            model.zero_grad()
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            y_pred = y_pred.to(device)
            #             print(y_pred.shape, y.shape)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        epoch_losses.append(np.mean(losses))

        fig = plt.figure(figsize=(14 ,5))
        ax1 = fig.add_subplot(121)
        plt.plot(epoch_losses)
        plt.title("Loss")

        validation_losses = []
        with torch.no_grad():
            for i, batch in enumerate(validation_loader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                y_pred = y_pred.to(device)


                loss = loss_func(y_pred, y)
                validation_losses.append(loss.item())
            epoch_losses_val.append(np.mean(losses))
    return epoch_losses, epoch_losses_val, model

def train(model, dataset, device, loss_func=None, lr=0.001, n_epoch=1000, batch_size=4, shuffle=True,
          validation_split=.15):
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Loading data")
    #     validation_split = .15
    shuffle_dataset = shuffle
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1,
                                                    sampler=valid_sampler)
    try:
        train_loss, val_loss, best_model = train_cycle(model, optimizer, loss_func, n_epoch, train_loader,
                                                       validation_loader, device)
    except KeyboardInterrupt:
        print("Keyboard interrupt, continue work")
        return
    except:
        print("Something went wrong")
        raise
    return train_loss, val_loss, best_model