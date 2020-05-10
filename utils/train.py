import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import traceback

#from model.generator import Generator
#from model.multiscale import MultiScaleDiscriminator
from .utils import get_commit_hash
from .validation import validate
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
import time
import itertools
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_epoch(model, iterator, optimizer, loss_func, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in tqdm(iterator):
        model.zero_grad()
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        y_pred = y_pred.to(device)
        #             print(y_pred.shape, y.shape)
        loss = loss_func(y_pred, y)
        acc = categorical_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def eval_epoch(model, iterator, loss_func, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            y_pred = y_pred.to(device)
            #             print(y_pred.shape, y.shape)
            loss = loss_func(y_pred, y)
            acc = categorical_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_cycle(model, optimizer, loss_func, n_epoch, train_loader, validation_loader, device):
    model.train()
    train_losses = []
    test_losses = []
    train_acces = []
    test_acces = []
    for epoch in range(n_epoch):
        start_time = time.time()
    
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_func, device)
        valid_loss, valid_acc = eval_epoch(model, validation_loader, loss_func, device)

        end_time = time.time()
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        train_acces.append(train_acc)
        test_acces.append(valid_acc)
#         print("Epoch: {}".format(epoch))
        
        
#         epoch_losses.append(np.mean(losses))

#         fig = plt.figure(figsize=(14 ,5))
#         ax1 = fig.add_subplot(121)
#         plt.plot(epoch_losses)
#         plt.title("Loss")

#         validation_losses = []
#         with torch.no_grad():
#             for i, batch in enumerate(validation_loader):
#                 x, y = batch
#                 x = x.to(device)
#                 y = y.to(device)

#                 y_pred = model(x)
#                 y_pred = y_pred.to(device)


#                 loss = loss_func(y_pred, y)
#                 validation_losses.append(loss.item())
#             epoch_losses_val.append(np.mean(losses))
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'audio-model.pt')
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%') 
    return epoch_losses, epoch_losses_val, model

def train(model, dataset, device, loss_func=None, lr=0.001, n_epoch=1000, batch_size=1, shuffle=True,
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


def train_depr(args, pt_dir, chkpt_path, trainloader, valloader, writer, logger, hp, hp_str):
    model_g = Generator(hp.audio.n_mel_channels).cuda()
    model_d = MultiScaleDiscriminator().cuda()

    optim_g = torch.optim.Adam(model_g.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.Adam(model_d.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))

    githash = get_commit_hash()

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    try:
        model_g.train()
        model_d.train()
        for epoch in itertools.count(init_epoch+1):
            if epoch % hp.log.validation_interval == 0:
                with torch.no_grad():
                    validate(hp, args, model_g, model_d, valloader, writer, step)

            trainloader.dataset.shuffle_mapping()
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
            for (melG, audioG), (melD, audioD) in loader:
                melG = melG.cuda()
                audioG = audioG.cuda()
                melD = melD.cuda()
                audioD = audioD.cuda()

                # generator
                optim_g.zero_grad()
                fake_audio = model_g(melG)[:, :, :hp.audio.segment_length]
                disc_fake = model_d(fake_audio)
                disc_real = model_d(audioG)
                loss_g = 0.0
                for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
                    loss_g += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                    for feat_f, feat_r in zip(feats_fake, feats_real):
                        loss_g += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))

                loss_g.backward()
                optim_g.step()

                # discriminator
                fake_audio = model_g(melD)[:, :, :hp.audio.segment_length]
                fake_audio = fake_audio.detach()
                loss_d_sum = 0.0
                for _ in range(hp.train.rep_discriminator):
                    optim_d.zero_grad()
                    disc_fake = model_d(fake_audio)
                    disc_real = model_d(audioD)
                    loss_d = 0.0
                    for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                        loss_d += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                        loss_d += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

                    loss_d.backward()
                    optim_d.step()
                    loss_d_sum += loss_d

                step += 1
                # logging
                loss_g = loss_g.item()
                loss_d_avg = loss_d_sum / hp.train.rep_discriminator
                loss_d_avg = loss_d_avg.item()
                if any([loss_g > 1e8, math.isnan(loss_g), loss_d_avg > 1e8, math.isnan(loss_d_avg)]):
                    logger.error("loss_g %.01f loss_d_avg %.01f at step %d!" % (loss_g, loss_d_avg, step))
                    raise Exception("Loss exploded")

                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss_g, loss_d_avg, step)
                    loader.set_description("g %.04f d %.04f | step %d" % (loss_g, loss_d_avg, step))

            if epoch % hp.log.save_interval == 0:
                save_path = os.path.join(pt_dir, '%s_%s_%04d.pt'
                    % (args.name, githash, epoch))
                torch.save({
                    'model_g': model_g.state_dict(),
                    'model_d': model_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
