import logging
import time
import argparse
import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing import List
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import pandas as pd

from classifier import FTVectorizer, FTClassifier, HP
from review_dataset import ReviewDataset



def load_review_dataset(filename):
    """ Download the date: list of texts with scores."""
    headers = ['polarity', 'text']
    examples = pd.read_csv(filename, encoding="utf-8", sep='\t', names=headers)
    # print distributions by rating or class
    # print(sentences.groupby('polarity').nunique())
    # return the list of rows : row = label and text
    return examples.text.to_list(), examples.polarity.to_list()


def eval_list(glabels, slabels):
    if (len(glabels) != len(slabels)):
        print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
        len(slabels), len(glabels)))
    n = min(len(slabels), len(glabels))
    incorrect_count = 0
    for i in range(0, n):
        if slabels[i] != glabels[i]: incorrect_count += 1
    acc = (n - incorrect_count) / n
    acc = acc * 100
    return acc


def ft_train(model, train_dataset, val_dataset, accelerator: str, devices: List[int]) -> pl.LightningModule:
    train_dataloader = DataLoader(train_dataset, batch_size=HP.batch_size, collate_fn=model.vectorizer.batch_collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=HP.batch_size, collate_fn=model.vectorizer.batch_collate_fn, shuffle=False)
    # Define early stopping for the training
    early_stop = EarlyStopping(monitor=HP.es_monitor, mode=HP.es_mode, min_delta=HP.es_min_delta, patience=HP.es_patience, verbose=True)
    # Define how the best model checkpoint is selected and saved
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=HP.ckpt_monitor, mode=HP.ckpt_mode)
    # trainer
    trainer = pl.Trainer(max_epochs=HP.max_epochs, callbacks=[early_stop, checkpoint_callback], log_every_n_steps=10, accelerator=accelerator, devices=devices)
    # run training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # training finished: recover the best model checkpoint
    best_model = FTClassifier.load_from_checkpoint(checkpoint_callback.best_model_path, vectorizer=model.vectorizer)
    return best_model


def ft_predict(model, texts: List[str], accelerator: str, devices: List[int]):
    dataset = ReviewDataset(texts, model.vectorizer)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=model.vectorizer.batch_collate_fn)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices)
    predicted_vect_list = trainer.predict(model, data_loader) # list of vectors of size 3
    predictions = torch.cat(predicted_vect_list, dim=0)  # matrix of dimension N x 3 with N=number of texts
    predicted_labels = model.vectorizer.devectorize_labels(predictions)
    return predicted_labels


# Run the project: it should run with no errors
def exec_project(accelerator: str = 'cuda', devices: List[int] = None, n_runs: int = 2):
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    datadir = "../data/"
    trainfile = datadir + "frdataset1_train.csv"
    devfile = datadir + "frdataset1_dev.csv"
    # testfile = datadir + "frdataset1_test.csv"
    testfile = None
    #
    vectorizer = FTVectorizer()
    # Load the train dataset
    train_texts, train_labels = load_review_dataset(trainfile)
    # fit the vectorizer using the training data
    vectorizer.fit(train_texts, train_labels)
    # create train dataset
    train_dataset = ReviewDataset(train_texts, vectorizer, train_labels)
    # create the validation (dev) dataset # correction par Guillaume
    val_texts, val_labels = load_review_dataset(devfile)
    val_dataset = ReviewDataset(val_texts, vectorizer, val_labels)
    if testfile is not None:
        test_texts, test_labels = load_review_dataset(testfile)
        test_dataset = ReviewDataset(test_texts, vectorizer, test_labels)
    else:
        test_dataset = test_labels = None

    print("-------> Hyperparameters values:")
    pprint(HP)
    devaccs = []
    testaccs = []
    for run_number in range(1, n_runs+1):
        print("============================= RUN: %s ==============================" % str(run_number))
        # Create the model
        model = FTClassifier(vectorizer)
        # train the model and get the best model version
        print(f"  {run_number}.1. Training the classifier (train data size = {len(train_texts)})...")
        model = ft_train(model, train_dataset, val_dataset, accelerator, devices)
        # evaluate on validation set
        print()
        print(f"  {run_number}.2. Evaluation on the dev dataset (dev data size = {len(val_texts)})...")
        pred_labels = ft_predict(model, val_dataset.texts, accelerator, devices)
        val_acc = eval_list(val_labels, pred_labels)
        devaccs.append(round(val_acc, 2))
        print("       ValAcc.: %.2f" % val_acc)
        if test_dataset is not None:
            # evaluate on test set
            pred_labels = ft_predict(model, test_dataset.texts, accelerator, devices)
            test_acc = eval_list(test_labels, pred_labels)
            testaccs.append(round(test_acc, 2))
            print("       TestAcc.: %.2f" % test_acc)
            # for text, pred_label, ref_label in zip(test_dataset.texts, pred_labels, test_labels):
            #     print("-->", text, " --- ", pred_label, " --- ", ref_label)
    print('\nCompleted %d runs.' % n_runs)
    return devaccs, testaccs




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-r', '--nruns', help='id (an integer) of the gpu to use. If not specified, model will run on cpu.', type=int, required=False, default=None)
    argparser.add_argument('-g', '--gpu_id', help='Number of runs (default: 1)', type=int, required=False, default=None)
    args = argparser.parse_args()
    n_runs = 1 if args.nruns is None else args.nruns
    if args.gpu_id is None :
        accelerator = 'cpu'
        devices = None
    else:
        accelerator = 'cuda'
        devices = [args.gpu_id]
    # run project
    seed_everything(123)
    start_time = time.perf_counter()
    devaccs, testaccs = exec_project(accelerator, devices, n_runs)
    total_exec_time = (time.perf_counter() - start_time)
    print("Dev accs:", devaccs)
    print("Test accs:", testaccs)
    print()
    if testaccs:
        print("Mean Dev Acc.: %.2f (%.2f)\tMean Test Acc.: %.2f (%.2f)" % (np.mean(devaccs), np.std(devaccs), np.mean(testaccs), np.std(testaccs)))
    else:
        print("Mean Dev Acc.: %.2f (%.2f)" % (
        np.mean(devaccs), np.std(devaccs)) )
    print("\nExec time: %.2f s. ( %d per run )" % (total_exec_time, total_exec_time / n_runs))

















