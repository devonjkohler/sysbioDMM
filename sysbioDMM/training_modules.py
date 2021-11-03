## Functions needed to train and test DMM

import numpy as np
import torch
from torch import nn, tensor

def process_minibatch(svi, train_data, epoch, which_mini_batch, shuffled_indices,
                      mini_batch_size, annealing_epochs, minimum_annealing_factor,
                      use_cuda=False):

    training_seq_lengths = tensor(np.array([len(train_data[i]) for i in range(len(train_data))]))
    training_data_sequences = tensor(np.array([train_data[i].values for i in range(len(train_data))]))
    N_train_data = len(training_seq_lengths)
    N_mini_batches = int(N_train_data / 10 + int(N_train_data % 10 > 0))


    if annealing_epochs > 0 and epoch < annealing_epochs:
        # compute the KL annealing factor appropriate
        # for the current mini-batch in the current epoch
        min_af = minimum_annealing_factor
        annealing_factor = min_af + (1.0 - min_af) * \
            (float(which_mini_batch + epoch * N_mini_batches + 1) /
             float(annealing_epochs * N_mini_batches))
    else:
        # by default the KL annealing factor is unity
        annealing_factor = 1.0

    # compute which sequences in the training set we should grab
    mini_batch_start = (which_mini_batch * mini_batch_size)
    mini_batch_end = np.min([(which_mini_batch + 1) * mini_batch_size,
                             N_train_data])
    mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
    # grab the fully prepped mini-batch using the helper function in the data loader
    mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
        = get_mini_batch(mini_batch_indices, training_data_sequences,
                         training_seq_lengths, cuda=use_cuda)
    mini_batch = mini_batch.float()
    mini_batch_reversed = mini_batch_reversed.float()
    mini_batch_mask = mini_batch_mask.float()

    # do an actual gradient step
    loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                    mini_batch_seq_lengths, annealing_factor)
    # keep track of the training loss
    return loss

def do_evaluation(dmm, svi, val_data, test_data, use_cuda):

    # Extract parameters form val and test data
    test_seq_lengths = tensor(np.array([len(val_data[i]) for i in range(len(val_data))]))
    test_data_sequences = tensor(np.array([val_data[i].values for i in range(len(val_data))]))
    val_seq_lengths = tensor(np.array([len(test_data[i]) for i in range(len(test_data))]))
    val_data_sequences = tensor(np.array([test_data[i].values for i in range(len(test_data))]))

    # package repeated copies of val/test data for faster evaluation
    # (i.e. set us up for vectorization)
    n_eval_samples = 1
    def rep(x):
        return np.repeat(x, n_eval_samples, axis=0)

    # get the validation/test data ready for the dmm: pack into sequences, etc.
    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = get_mini_batch(
        np.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
        val_seq_lengths, cuda=use_cuda)
    val_batch = val_batch.float()
    val_batch_reversed = val_batch_reversed.float()
    val_batch_mask = val_batch_mask.float()

    test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = \
        get_mini_batch(np.arange(n_eval_samples * test_data_sequences.shape[0]),
                       rep(test_data_sequences),
                       test_seq_lengths, cuda=use_cuda)
    test_batch = test_batch.float()
    test_batch_reversed = test_batch_reversed.float()
    test_batch_mask = test_batch_mask.float()

    # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
    dmm.rnn.eval()

    # compute the validation and test loss
    val_nll = svi.evaluate_loss(val_batch, val_batch_reversed, val_batch_mask,
                                val_seq_lengths) / sum(val_seq_lengths)
    test_nll = svi.evaluate_loss(test_batch, test_batch_reversed, test_batch_mask,
                                 test_seq_lengths) / sum(test_seq_lengths)

    # put the RNN back into training mode (i.e. turn on drop-out if applicable)
    dmm.rnn.train()
    return val_nll, test_nll

def get_mini_batch(mini_batch_indices, sequences, seq_lengths, cuda=False):
    # get the sequence lengths of the mini-batch
    seq_lengths = seq_lengths[mini_batch_indices]
    # sort the sequence lengths
    _, sorted_seq_length_indices = torch.sort(seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

    # compute the length of the longest sequence in the mini-batch
    T_max = torch.max(seq_lengths)
    # this is the sorted mini-batch
    mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
    # this is the sorted mini-batch in reverse temporal order
    mini_batch_reversed = reverse_sequences(mini_batch, sorted_seq_lengths)
    # get mask for mini-batch
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)

    # cuda() here because need to cuda() before packing
    if cuda:
        mini_batch = mini_batch.cuda()
        mini_batch_mask = mini_batch_mask.cuda()
        mini_batch_reversed = mini_batch_reversed.cuda()

    # do sequence packing
    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(
        mini_batch_reversed, sorted_seq_lengths, batch_first=True
    )

    return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths

## Helper functions from pyro that are not currently installed in the new version
# this function takes the hidden state as output by the PyTorch rnn and
# unpacks it it; it also reverses each sequence temporally
def pad_and_reverse(rnn_output, seq_lengths):

    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output

def reverse_sequences(mini_batch, seq_lengths):

    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch

# this function returns a 0/1 mask that can be used to mask out a mini-batch
# composed of sequences of length `seq_lengths`

def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0 : seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask