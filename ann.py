"""
ann.py
------

A small feedforward neural network used as the bee's controller.
The weights of the NN are the "genome" that the ES evolves.

Design choices:
- Single hidden layer: the lecture specifies to keep the network as small as possible to have fewer local minima and have faster learning.
- Small number of neurons in the hidden layer: 4 neurons
- Tanh activation everywhere: bounded outputs in [-1, 1], which
    matches Bee.step()'s expected input range directly

This is a REACTIVE controller: action = f(observation), no memory.
That's a deliberate Day 2 scope choice (see AE4350 ER lecture on
reactive vs pro-active controllers) -- it's the simplest thing that
could possibly work, and a natural "complexity" extension later is to
add a recurrent hidden state (pro-active controller: a = f(i, s),
s = g(i, s)), which only requires feeding the previous hidden
activation back in as an extra input. (So we start with no memory -> we might add memory afterwards)

The whole point of `get_flat_weights` / `set_flat_weights` is that the
ES doesn't need to know anything about layers, matrices, or neurons --
it only ever manipulates ONE flat real-valued vector (the genome). This
keeps the ES and the network architecture completely decoupled so I can modify the neural network architecture later.

"""

import numpy as np

class FeedforwardNN:
    def __init__(self, n_inputs = 2, n_hidden = 4, n_outputs = 2, seed = None):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        rng = np.random.default_rng(seed)
 
        # Small random init (does not matter much -- the ES will
        # immediately start overwriting these via set_flat_weights,
        # but we still want a valid starting point).
        self.W1 = rng.normal(0, 0.5, size=(n_inputs, n_hidden))
        self.b1 = np.zeros(n_hidden) #bias
        self.W2 = rng.normal(0, 0.5, size=(n_hidden, n_outputs))
        self.b2 = np.zeros(n_outputs) #bias

    @property
    def num_weights(self):
        """Total number of weigths in the network (genome length for the ES)"""

        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def get_flat_weights(self):
        """Return a 1D array of all weights in the network, in a fixed order.
        This is the "genome" that the ES will evolve."""
        return np.concatenate([self.W1.flatten(), self.b1.flatten(),
                               self.W2.flatten(), self.b2.flatten()])

    def set_flat_weights(self, flat):
        """Unpack the flat genome vector back into W1, b1, W2, b2"""

        i = 0
        n = self.W1.size
        self.W1 = flat[i:i+n].reshape(self.W1.shape); i += n
        n = self.b1.size
        self.b1 = flat[i:i+n].reshape(self.b1.shape); i += n
        n = self.W2.size
        self.W2 = flat[i:i+n].reshape(self.W2.shape); i += n
        n = self.b2.size
        self.b2 = flat[i:i+n].reshape(self.b2.shape); i += n

    def forward(self, obs):
        """Forward pass through the network, returning the output vector.
        obs: 1D array of length n_inputs (n_inputs, )
        returns: 1D array of length n_outputs, each in [-1,1] (n_outputs,)"""

        # Input -> hidden layer
        h = np.tanh(obs @ self.W1 + self.b1)

        # Hidden layer -> output
        out = np.tanh(h @ self.W2 + self.b2)

        return out

    def make_controller (n_inputs = 2, n_hidden = 4, n_outputs = 2, seed = None):
        """Factory function to create a new FeedforwardNN instance.
        This is the function that the ES will call to create a new
        controller for each individual in the population."""
        return FeedforwardNN(n_inputs, n_hidden, n_outputs, seed=seed)


    