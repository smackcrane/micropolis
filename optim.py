import numpy as np
import bisect
import functional as F

class MH:
    """Full-batch Metropolis-Hastings optimizer"""

    def __init__(self, model, lr=1, T=1, test=None):
        # test parameter is unused, included for compatibility
        self.model = model
        self.lr = lr
        self.T = T
        # we assume the same full batch is used at each iteration
        # so we can store the sample loss for efficiency
        self.sample_nll = 1e5

        self.rng = np.random.default_rng()

    def step(self, X, y, batch_size=None):
        """Step the optimizer: propose a sample, test, accept or reject

        batch_size parameter is unused, included for compatibility
        """

        # propose new sample
        proposal = self.model.copy()
        for p in proposal.parameters():
            p += self.lr * self.rng.normal(size=p.shape)

        # compute loss
        y_pred = proposal(X)
        proposal_nll = F.negative_log_likelihood(y_pred, y)

        # acceptance test
        ratio = np.exp((self.sample_nll - proposal_nll) / self.T)
        u = self.rng.uniform(0,1)
        accept = u < ratio

        if accept:
            self.model.set_parameters(proposal)
            self.sample_nll = proposal_nll

        return {
                'accept' : accept,
                'proposal_nll' : proposal_nll,
                'ratio' : ratio,
                }

    def __repr__(self):
        return f'MH(lr={self.lr:.0e}, T={self.T:.0e})'

class MMH:
    """Minibatch Metropolis-Hastings optimizer

    lr: float, learning rate
    T: float, temperature (must be > 0 if test='exact')
    test: 'naive' for standard MH test
          'Barker' for approximate Barker test cf. 6.3 of https://www.jmlr.org/papers/volume18/15-205/15-205.pdf
          'exact' for exact Barker test as in https://arxiv.org/abs/1610.06848
    """

    def __init__(self, model, lr=1, T=1, test='naive'):
        self.model = model
        self.lr = lr
        self.T = T
        self.test = test

        if test == 'naive': self.step = self._naive_step
        elif test == 'Barker': self.step = self._Barker_step
        elif test == 'exact':
            assert T > 0, f'test="{test}" requires T > 0, got {T=}'
            self.step = self._exact_step
            # load X_correction distribution
            xcorr = np.genfromtxt('data/norm2log4000_20_1.0.txt') # stored as density
            xcorr[:,1] = np.cumsum(xcorr[:,1]) # convert to cdf
            xcorr[:,1] = xcorr[:,1] / xcorr[-1,1] # make total probability equal to 1
            self.X_correction_cdf = xcorr
        else: raise ValueError(f'test="{test}" not recognized')

        self.rng = np.random.default_rng()

    def propose(self):
        """Propose a new set of parameters"""
        proposal = self.model.copy()
        for p in proposal.parameters():
            p += self.lr * self.rng.normal(size=p.shape)
        return proposal

    def draw_minibatch(self, X, y, batch_size):
        """Randomly draw a minibatch from training data"""
        idx = self.rng.integers(low=0, high=len(y), size=batch_size)
        return X[idx], y[idx]

    def compute_logratio(self, proposal, X_batch, y_batch, reduction):
        """Compute log likelihood ratio between proposal and current sample"""
        y_proposal = proposal(X_batch)
        y_sample = self.model(X_batch)
        proposal_nll = F.negative_log_likelihood(y_proposal, y_batch, reduction=reduction)
        sample_nll = F.negative_log_likelihood(y_sample, y_batch, reduction=reduction)
        return sample_nll - proposal_nll

    def _naive_step(self, X, y, batch_size):
        """Step the optimizer using naive minibatch test"""

        proposal = self.propose()
        X_batch, y_batch = self.draw_minibatch(X, y, batch_size)
        logratio = self.compute_logratio(proposal, X_batch, y_batch, reduction='mean')

        # acceptance test
        ratio = np.exp(logratio / self.T)
        u = self.rng.uniform(0,1)
        accept = u < ratio

        if accept:
            self.model.set_parameters(proposal)

        return {
                'accept' : accept,
                'logratio' : logratio,
                }

    def _Barker_step(self, X, y, batch_size):
        """Step the optimizer with approximate Barker acceptance test"""

        proposal = self.propose()
        X_batch, y_batch = self.draw_minibatch(X, y, batch_size)
        logratio = self.compute_logratio(proposal, X_batch, y_batch, reduction='mean')

        # add random noise if temperature is nonzero
        if self.T > 0:
            X_normal = self.rng.normal(loc=0, scale=1)
            logratio = logratio / self.T + X_normal

        # acceptance test
        accept = logratio > 0

        if accept:
            self.model.set_parameters(proposal)

        return {
                'accept' : accept,
                'logratio' : logratio,
                }

    def _exact_step(self, X, y, batch_size):
        """Step the optimizer using corrected Barker minibatch test"""

        proposal = self.propose()

        # loop to enlarge minibatch until variance condition is satisfied
        batch_total = 0
        logratio_batch = np.empty(shape=(0,)) # container for log ratios
        sample_variance = 1e5
        target_variance = 1 # this is fixed to match the pre-computed X_correction distribution
        while sample_variance > target_variance:

            # enlarge minibatch
            # NB. we make no effort to avoid duplicates---they should be rare enough to have no effect for small batch sizes
            batch_total += batch_size
            X_new, y_new = self.draw_minibatch(X, y, batch_size)

            # compute log ratios for new samples and append to list
            logratios = self.compute_logratio(proposal, X_new, y_new, reduction='none')
            logratios /= self.T
            logratio_batch = np.concatenate((logratio_batch, logratios))

            # compute sample variance---N.B. not simply variance, but rather
            #   estimated variance of sampling distribution of mean, so `/ batch_total` by CLT
            sample_variance = np.var(logratio_batch) / batch_total

            # check if we've reached full batch
            if batch_total >= len(y):
                print('\nWARNING: full batch consumed without satisfying variance condition')
                break # if we reach full batch, ignore variance condition and proceed

        logratio = np.mean(logratio_batch)

        # acceptance test
        if sample_variance > target_variance:   # if we consumed the full batch and broke the loop,
            accept = logratio > 0               # just test if log ratio is positive
        else:
            X_normal = self.rng.normal(loc=0, scale=np.sqrt(target_variance-sample_variance))
            X_correction = self.sample_X_correction()
            statistic = logratio + X_normal + X_correction
            accept = statistic > 0

        if accept:
            self.model.set_parameters(proposal)

        return {
                'accept' : accept,
                'logratio' : logratio,
                'batch_total' : batch_total,
                'sample_variance' : sample_variance,
                }

    def sample_X_correction(self):
        """Sample from X_correction distribution

        sample u ~ U(0,1) uniform, then find the corresponding location in correction CDF
        """
        u = self.rng.uniform(0,1)

        # use [:-1,---] because bisect_left(a, u) goes up to len(a) (which would then IndexError)
        idx = bisect.bisect_left(self.X_correction_cdf[:-1,1], u)
        return self.X_correction_cdf[idx, 0]

    def __repr__(self):
        return f'MMH(lr={self.lr:.0e}, T={self.T:.0e}, test="{self.test}")'

