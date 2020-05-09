import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import itertools
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight

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

emotion_dict = {'ang': 0,
                'dis': 1,
                'hap': 2,
                'sad': 3,
                'sca': 4,
                'sur': 5,
                'neu': 6
                }

emo_keys = list(['ang', 'hap', 'sad', 'fea', 'sur', 'neu', 'sca'])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels

def display_results(y_test, pred_probs, cm=True):
    pred = np.argmax(pred_probs, axis=-1)
    one_hot_true = one_hot_encoder(y_test, len(pred), len(emotion_dict))
    print('Test Set Accuracy =  {0:.3f}'.format(accuracy_score(y_test, pred)))
    print('Test Set F-score =  {0:.3f}'.format(f1_score(y_test, pred, average='macro')))
    print('Test Set Precision =  {0:.3f}'.format(precision_score(y_test, pred, average='macro')))
    print('Test Set Recall =  {0:.3f}'.format(recall_score(y_test, pred, average='macro')))
    if cm:
        plot_confusion_matrix(confusion_matrix(y_test, pred), classes=emo_keys)