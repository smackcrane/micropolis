import numpy as np
from collections import namedtuple
import functional as F

class MH:
    """Metropolis-Hastings optimizer"""

    def __init__(self, model, lr=1, T=1):
        self.model = model
        self.lr = lr
        self.T = T
        self.sample_nll = 1e5

        self.rng = np.random.default_rng()
        self.Stats = namedtuple('Stats', ['accept', 'proposal_nll', 'ratio', 'p'])

    def step(self, X, y):
        """Step the optimizer: propose a sample, test, accept or reject"""

        # propose new sample
        proposal = self.model.copy()
        for p in proposal.parameters():
            p += self.lr * self.rng.normal(size=p.shape)

        # compute loss
        y_pred = proposal(X)
        proposal_nll = F.negative_log_likelihood(y_pred, y, T=self.T)

        # acceptance test
        ratio = np.exp(self.sample_nll - proposal_nll)
        p = self.rng.uniform(0,1)
        accept = p < ratio

        if accept:
            self.model.set_parameters(proposal)
            self.sample_nll = proposal_nll

        return self.Stats(accept, proposal_nll, ratio, p)
