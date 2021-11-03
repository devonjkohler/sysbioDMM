## Core functions for building layers of DMM
## code adapted from https://pyro.ai/examples/dmm.html

import torch
from torch import nn

class Emitter(nn.Module):
    """
    Parameterizes the linear observation likelihood p(x_t | z_t).
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()

        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)

        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus() ## To ensure positive output

        ## NOTE: These might be useful depending on how we sample p(x_t|z_t)
        # self.sigmoid = nn.Sigmoid()
        # self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        means `loc_p` and variance `scale_p` that parameterizes the normal distribution p(x_t|z_t)
        """

        # compute the gating function
        _gate = self.relu(self.lin_z_to_hidden(z_t))
        gate = self.sigmoid(self.lin_hidden_to_input(_gate))

        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_z_to_hidden(z_t))
        proposed_mean = self.lin_hidden_to_input(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes
        # a linear transformation of z_{t-1} with the proposed mean
        # modulated by the gating function
        loc_p = (1 - gate) * self.lin_hidden_to_input(z_t) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed
        # mean from above as input. the softplus ensures that scale is positive
        scale_p = self.softplus(self.linear(self.relu(proposed_mean)))

        return loc_p, scale_p

class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    """
    def __init__(self, z_dim, transition_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)

        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent z_{t-1} corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1})
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = self.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes
        # a linear transformation of z_{t-1} with the proposed mean
        # modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed
        # mean from above as input. the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scaleb

class Combiner(nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below).
    The guide is used to approximate the posterior p(z_{1:T}| x_{1:T}). In training
    we use the data `D` to estimate p(z^1_{1:T}, z^2_{1:T}, .., z^N_{1:T}|D).
    """
    def __init__(self, z_dim, rnn_dim):
        super().__init__()

        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        #self.softplus = nn.ReLU()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(x_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, x_{t:T})
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        #scale = self.softplus(h_combined)

        # return loc, scale which can be fed into Normal
        return loc, scale