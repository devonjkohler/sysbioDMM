## DMM of the Lac Operon, trained with simulated data

import pickle
import time
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from torch import tensor
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO

from sysbioDMM import DMM, process_minibatch, do_evaluation

## Load data
with open(r"data/sim_data.pickle", "rb") as input_file:
    sim_data = pickle.load(input_file)

## Sample observations at step=100
sim_data_samples = [sim_data[i].iloc[::100, :].drop('time', axis=1) for i in range(len(sim_data))]

## Split into training, val, test sets
random.seed(2)
train_idx = random.sample(range(len(sim_data_samples)), int(len(sim_data_samples) * .8))
val_test_idx = [x for x in range(len(sim_data_samples)) if x not in train_idx]
val_idx = random.sample(val_test_idx, int(len(val_test_idx) * .5))
test_idx = [x for x in val_test_idx if x not in val_idx]

train_data = [sim_data_samples[i] for i in train_idx]
val_data = [sim_data_samples[i] for i in val_idx]
test_data = [sim_data_samples[i] for i in test_idx]

## Normalize input
long_training = pd.concat(train_data)
train_mean = long_training.mean(axis = 0)
train_std = long_training.std(axis = 0)

for i in range(len(train_data)):
    for col in range(len(train_data[i].columns)):
        train_data[i].iloc[:, col] = (train_data[i].iloc[:, col] - train_mean[col]) / train_std[col]
        if i < len(val_data):
            val_data[i].iloc[:, col] = (val_data[i].iloc[:, col] - train_mean[col]) / train_std[col]
            test_data[i].iloc[:, col] = (test_data[i].iloc[:, col] - train_mean[col]) / train_std[col]

    ## Data simulated without variation in dna_Lac_Operon or Biomass. Just drop them temporarily
    train_data[i] = train_data[i].drop(columns=['dna_Lac_Operon', 'Biomass'])
    if i < len(val_data):
        val_data[i] = val_data[i].drop(columns=['dna_Lac_Operon', 'Biomass'])
        test_data[i] = test_data[i].drop(columns=['dna_Lac_Operon', 'Biomass'])


## Setup model
dmm = DMM(input_dim=9, z_dim=50, emission_dim=50,
          transition_dim=75, rnn_dim=250, rnn_dropout_rate=0.4,
          rnn_layers=1, num_iafs=0, iaf_dim=50, use_cuda=False)

# setup optimizer
adam_params = {"lr": .05, "betas": (.9, .999),
               "clip_norm": 1., "lrd": 0.001,
               "weight_decay": 0.001}
optimizer = ClippedAdam(adam_params)

svi = SVI(dmm.model, dmm.guide, optimizer, Trace_ELBO())

## Train/validate model
training_seq_lengths = tensor(np.array([len(train_data[i]) for i in range(len(train_data))]))
training_data_sequences = tensor(np.array([train_data[i].values for i in range(len(train_data))]))
N_train_data = len(training_seq_lengths)
N_mini_batches = int(N_train_data / 10 + int(N_train_data % 10 > 0))
N_train_time_slices = sum(training_seq_lengths)

num_epochs = 501 ## Temp small test epochs
val_test_frequency = 25

times = [time.time()]
val_nll_list = list()
for epoch in range(num_epochs):
    # accumulator for our estimate of the negative log likelihood
    # (or rather -elbo) for this epoch
    epoch_nll = 0.0
    # prepare mini-batch subsampling indices for this epoch
    shuffled_indices = np.arange(N_train_data)
    np.random.shuffle(shuffled_indices)

    # process each mini-batch; this is where we take gradient steps
    for which_mini_batch in range(N_mini_batches):
        epoch_nll += process_minibatch(svi, train_data, epoch, which_mini_batch, shuffled_indices,
                                       10, 100, .01, use_cuda=False)

    # report training diagnostics
    times.append(time.time())
    epoch_time = times[-1] - times[-2]
    print("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
          (epoch, epoch_nll / N_train_time_slices, epoch_time))

    # do evaluation on test and validation data and report results
    if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
        val_nll, test_nll = do_evaluation(dmm, svi, val_data, test_data, use_cuda=False)
        val_nll_list.append(val_nll)
        print(
            "[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll)
        )


