"""
Train a classifier to feed into the agent's memory.

By default, this uses the MixMatch semi-supervised
learning algorithm to improve data efficiency, since there
are many more frames in the demonstrations than there are
in the labeled dataset.
"""

import itertools
from multiprocessing.dummy import Pool
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from constants import NUM_LABELS
from model import StateClassifier
from util import Augmentation, atomic_save, mirror_obs
from sklearn.model_selection import train_test_split
from tqdm import trange


LR = 1e-4
BATCH = 128
USE_MIXMATCH = True
NUM_AUGMENTATIONS = 2
UNLABELED_WEIGHT = 1
MIXUP_ALPHA = 0.75
TEMPERATURE = 0.5


def main():
    model = StateClassifier()
    if os.path.exists('save_classifier.pkl'):
        model.load_state_dict(torch.load('save_classifier.pkl'))
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train, test = load_labeled_images()
    recordings = load_unlabeled_images()
    thread_pool = Pool(8)
    for i in itertools.count():
        test_loss = classification_loss(thread_pool, model, test).item()
        if USE_MIXMATCH:
            loss = mixmatch_loss(model,
                                 *labeled_data(thread_pool, model, train),
                                 *unlabeled_data(thread_pool, model, recordings))
            print('step %d: test=%f mixmatch=%f' % (i, test_loss, loss.item()))
        else:
            loss = classification_loss(thread_pool, model, train)
            print('step %d: test=%f train=%f' % (i, test_loss, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % 100:
            atomic_save(model.state_dict(), 'save_classifier.pkl')


def load_labeled_images():
    labeled_data_path = '../crowdsource_data/labeled_data'

    images, labels = [], []

    folders = os.listdir(labeled_data_path)
    for folder in folders:
        folder_path = os.path.join(labeled_data_path, folder)
        item_names = os.listdir(folder_path)
        print('read', folder_path)
        for i in trange(len(item_names)):
            item_name = item_names[i]
            item_path = os.path.join(folder_path, item_name)
            loaded = np.load(item_path)
            short_format_label, image = loaded['labels'], loaded['image']
            image = (image * 255).astype(np.uint8)
            assert len(short_format_label), 'empty label'

            try:
                label = np.zeros(NUM_LABELS)
                label[short_format_label] = 1.0
                images.append(image)
                labels.append(label)
            except IndexError:
                print(item_path, short_format_label)

    images = np.asarray(images)
    labels = np.asarray(labels).astype(np.float32)

    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.1, random_state=42
    )

    return (images_train, labels_train), (images_test, labels_test)


def load_unlabeled_images():
    unlabeled_data_path = '../crowdsource_data/unlabeled_data'

    images = []

    item_names = os.listdir(unlabeled_data_path)
    print('read', unlabeled_data_path)
    for i in trange(len(item_names)):
        item_name = item_names[i]
        item_path = os.path.join(unlabeled_data_path, item_name)
        loaded = np.load(item_path)
        image = loaded['image']
        image = (image * 255).astype(np.uint8)
        images.append(image)

    images = np.asarray(images)

    return torch.from_numpy(images)


def classification_loss(pool, model, dataset):
    image_tensor, label_tensor = labeled_data(pool, model, dataset)
    logits = model(image_tensor)
    loss = nn.BCEWithLogitsLoss()
    return loss(logits, label_tensor)


def mixmatch_loss(model, real_images, real_labels, other_images, other_labels):
    real_images, real_labels, other_images, other_labels = mixmatch(real_images, real_labels,
                                                                    other_images, other_labels)
    model_out = model(torch.cat([real_images, other_images]))
    real_out = model_out[:real_images.shape[0]]
    other_out = model_out[real_images.shape[0]:]

    bce = nn.BCEWithLogitsLoss()
    real_loss = bce(real_out, real_labels)
    other_loss = torch.mean(torch.pow(torch.sigmoid(other_out) - other_labels, 2))
    return real_loss + UNLABELED_WEIGHT * other_loss


def mixmatch(real_images, real_labels, other_images, other_labels):
    all_images = torch.cat([real_images, other_images])
    all_labels = torch.cat([real_labels, other_labels])
    indices = list(range(all_images.shape[0]))
    random.shuffle(indices)
    all_images, all_labels = mixup(all_images, all_labels,
                                   all_images[indices], all_labels[indices])
    return (all_images[:real_images.shape[0]], all_labels[:real_labels.shape[0]],
            all_images[real_images.shape[0]:], all_labels[real_labels.shape[0]:])


def mixup(real_images, real_labels, other_images, other_labels):
    probs = []
    for _ in range(real_images.shape[0]):
        p = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        probs.append(min(p, 1 - p))
    prob_tensor = torch.from_numpy(np.array(probs, dtype=np.float32)).to(real_images.device)
    interp_images = (real_images.float() + prob_tensor.view(-1, 1, 1, 1)
                     * (other_images - real_images).float()).byte()
    interp_labels = real_labels + prob_tensor.view(-1, 1) * (other_labels - real_labels)
    return interp_images, interp_labels


def labeled_data(pool, model, dataset):
    def augment_image(image):
        aug = Augmentation()
        img = np.array(aug.apply(image))
        if random.random() < 0.5:
            img = mirror_obs(img)
        return img

    images, labels = dataset
    idx = np.random.choice(len(images), BATCH)

    images = pool.map(augment_image, images[idx])
    images = np.asarray(images)
    labels = labels[idx]

    image_tensor = model_tensor(model, images)
    label_tensor = model_tensor(model, labels)
    return image_tensor, label_tensor


def unlabeled_data(pool, model, images):
    def augment_image(image):
        img_set = []
        for _ in range(NUM_AUGMENTATIONS):
            aug = Augmentation()
            img1 = np.array(aug.apply(image))
            if random.random() < 0.5:
                img1 = mirror_obs(img1)
            img_set.append(img1)
        return img_set

    idx = np.random.choice(len(images), BATCH)
    images = pool.map(augment_image, images[idx])

    flat_list = [item for sublist in images for item in sublist]
    images = np.asarray(flat_list)

    image_tensor = model_tensor(model, images)
    preds = torch.sigmoid(model(image_tensor)).detach()
    preds = preds.view(BATCH, NUM_AUGMENTATIONS, NUM_LABELS)
    mixed = torch.mean(preds, dim=1, keepdim=True)
    sharpened = sharpen_predictions(mixed)
    broadcasted = (sharpened + torch.zeros_like(preds)).view(-1, NUM_LABELS)
    return image_tensor, broadcasted


def sharpen_predictions(preds):
    pow1 = torch.pow(preds, 1 / TEMPERATURE)
    pow2 = torch.pow(1 - preds, 1 / TEMPERATURE)
    return pow1 / (pow1 + pow2)


def model_tensor(model, ndarray):
    device = next(model.parameters()).device
    return torch.from_numpy(ndarray).to(device)


if __name__ == '__main__':
   # train, test = load_labeled_images()
   # thread_pool = Pool(8)
   # model = StateClassifier()
   # print(train[1].dtype)
   # it, lt = labeled_data(thread_pool, model, train)
   # print(it.dtype, lt.dtype)
    main()
