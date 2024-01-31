import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nn
import optim
import functional as F

# suppress overflow in exp warning
import warnings
warnings.filterwarnings('ignore', message='overflow encountered in exp')

rng = np.random.default_rng()

# make up a dataset
from sklearn.datasets import make_moons
n_samples = 500
noise = 0.1
print(f'making moons with {n_samples=}, {noise=} ... ', end='', flush=True)
X, y = make_moons(n_samples=n_samples, noise=noise)
print('done', flush=True)

# initalize model
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10,10),
    nn.ReLU(),
    nn.Linear(10,2),
)
print()
print(model)
print()

initial_model = model.copy()

# plot decision boundary
def plot_decision_boundary(X, y, model, ax, mesh_size=0.025, title=''):
    """Plot model decision boundary along with dataset"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # x-limits
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # y-limits
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_size),
                         np.arange(y_min, y_max, mesh_size))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    scores = list(model(Xmesh))
    Z = np.array([x0 > x1 for (x0, x1) in scores])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)

# train model

epochs = 2_000
sample_nll = 1e5 # negative log likelihood
lr = 1e-1 # learning rate
T = 3e0 / n_samples # temperature
optimizer = optim.MH(model, lr=lr, T=T)
verbose = True
train_accepted = []
train_sample_nll = []
train_proposal_nll = []
train_lr = []
best_nll = 1e5
best_model = model.copy()

for epoch in range(epochs):

    # step optimizer
    stats = optimizer.step(X, y)
    accept, proposal_nll, ratio = stats['accept'], stats['proposal_nll'], stats['ratio']

    # record and print stats
    if accept:
        sample_nll = proposal_nll
    train_accepted.append(accept)
    train_sample_nll.append(sample_nll)
    train_proposal_nll.append(proposal_nll)
    if verbose:
        outcome = 'Accept' if accept else 'Reject'
        print(f'\rEp {epoch+1:6d}/{epochs}  {outcome}    sample: {sample_nll:3.3f}  proposal: {proposal_nll:3.3f}    {ratio=:.2f}                           '[:100], end='')
    # record best model
    if sample_nll < best_nll:
        best_nll = sample_nll
        best_model = model.copy()

# training results
print()
print(f'Total log likelihood: {sample_nll:.3f}    Probability: {np.exp(-sample_nll):.3f}    (with {T=:.1e})')
print(f'Acceptance rate: {np.sum(train_accepted) / len(train_accepted):.1%}')

# plots
gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,:])
plot_decision_boundary(X, y, initial_model, ax1, title="Initialized model")
plot_decision_boundary(X, y, best_model, ax2, title=f'Best model after {epochs} epochs')
ax3.plot(train_proposal_nll, label='proposal', c='orange')
ax3.plot(train_sample_nll, label='sample', c='blue')
ax3.set_title(f'Sample and Proposal Loss')
ax3.set_ylabel(f'negative log likelihood, {T=:.1e}')
ax3.set_ylim(0,50/n_samples)
ax3.legend()
plt.tight_layout()
plt.show()

