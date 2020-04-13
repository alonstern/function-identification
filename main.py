import argparse
import datetime
import sys

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from torch.utils import data

from dataset import FunctionIdentificationDataset
from model import CNNModel

torch.manual_seed(1)

kernel_size = 20


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("dataset_path", help="Path to the directory with the binaries for the dataset "
                                                      "(e.g ~/security.ece.cmu.edu/byteweight/elf_32")
    args = argument_parser.parse_args()

    print("Preprocessing")
    dataset = FunctionIdentificationDataset(args.dataset_path, block_size=1000, padding_size=kernel_size - 1)

    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

    model = CNNModel(embedding_dim=64, vocab_size=258, hidden_dim=16, tagset_size=2, kernel_size=kernel_size)

    print("Training")
    train_model(model, train_dataset)

    print("Testing")
    test_model(model, test_dataset)


def test_model(model, test_dataset):
    test_loader = data.DataLoader(test_dataset)
    model.eval()
    with torch.no_grad():
        all_tags = []
        all_tag_scores = []
        for sample, tags in tqdm.tqdm(test_loader):
            sample = sample[0]
            tags = tags[0]

            tag_scores = model(sample)

            all_tags.extend(tags.numpy())
            all_tag_scores.extend(tag_scores.numpy())

        all_tags = numpy.array(all_tags)
        all_tag_scores = numpy.array(all_tag_scores).argmax(axis=1)
        accuracy = accuracy_score(all_tags, all_tag_scores)
        pr = precision_score(all_tags, all_tag_scores)
        recall = recall_score(all_tags, all_tag_scores)
        f1 = f1_score(all_tags, all_tag_scores)

        print("accuracy: {}".format(accuracy))
        print("pr: {}".format(pr))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))


def train_model(model, train_dataset):
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = data.DataLoader(train_dataset, shuffle=True)
    model.train()
    for sample, tags in tqdm.tqdm(train_loader):
        sample = sample[0]
        tags = tags[0]
        model.zero_grad()

        tag_scores = model(sample)

        loss = loss_function(tag_scores, tags)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
