import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import nn
import optim
import functional as F

# suppress overflow in exp warning
import warnings
warnings.filterwarnings('ignore', message='overflow encountered in exp')
warnings.filterwarnings('ignore', message='divide by zero encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')

rng = np.random.default_rng()

# load dataset
try: # if we have it saved
    data = pd.read_csv('optical_recognition_of_handwritten_digits.csv')
    X = data.drop(columns='class').astype(float)
    y = data['class']
    data_url = 'https://archive.ics.uci.edu/static/public/80/data.csv'
except FileNotFoundError: # if not saved, download it
    from ucimlrepo import fetch_ucirepo 
    optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) # fetch dataset 
    # data (as pandas dataframes) 
    X = optical_recognition_of_handwritten_digits.data.features.astype(float)
    y = optical_recognition_of_handwritten_digits.data.targets
    data_url = optical_recognition_of_handwritten_digits.metadata['data_url']

X = X.to_numpy()
y = y.to_numpy(dtype=np.intc).squeeze() # in download case, y has an extra dimension of size 1

test_size=0.3
Xy_splits = train_test_split(X, y, test_size=test_size, stratify=y)
X_train, X_val, y_train, y_val = Xy_splits

print(f'Loaded UCI handwritten digits dataset, obtained from {data_url}')
print(f'Number of samples: {len(y)}')
print(f'Validation split: {test_size}    train samples: {len(y_train)}    val samples: {len(y_val)}')

# visualize some samples

#fig, axs = plt.subplots(nrows=3, ncols=8, figsize=(16,6))
#for i, ax in enumerate(np.ravel(axs)):
#    ax.imshow(np.array(X[i]).reshape((8,8)), cmap='grey')
#    ax.axis(False)
#plt.tight_layout()
#plt.show()

# initalize model
model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,10),
)
print()
print(model)
print()

initial_model = model.copy()

def train(model, optimizer, Xy_splits, batch_size, iterations, val_freq=100, time_limit=None, verbose=True, label=''):
    """Perform a training run and return history.

    Runs for specified number of iterations or specified time limit, whichever is shorter
    Xy_splits : (X_train, X_val, y_train, y_val)
    """

    X_train, X_val, y_train, y_val = Xy_splits
    if not batch_size:
        batch_size = len(y_train)

    t0 = time.time()
    # training loop setup
    accepted = []
    samples_consumed = [] # samples per iteration
    ms_consumed = [] # milliseconds per val_freq iterations 
    val_loss = []
    val_acc = []
    best_nll = 1e5
    best_model = model.copy()

    if verbose:
        print('#'*20 + ' TRAINING START ' + '#'*20)
        print(f'{iterations=}  {batch_size=}  {time_limit=}s')
        print(optimizer)
        print()

    # training loop
    t = time.time()
    for i in range(iterations+1):
        
        # step optimizer
        stats = optimizer.step(X_train, y_train, batch_size)
        accepted.append(stats['accept'])
        samples_consumed.append(stats.get('batch_total', batch_size))

        # validation
        if i % val_freq == 0:
            ms_consumed.append(1000*(time.time() - t))
            y_pred = model(X_val)
            val_nll = F.negative_log_likelihood(y_pred, y_val)
            val_acc_score = accuracy_score(y_val, np.argmax(y_pred, axis=1, keepdims=False))
            val_loss.append(val_nll)
            val_acc.append(val_acc_score)

            # print stats
            accept_rate = np.mean(accepted)
            if verbose:
                s = f'\rIter {i:6d}/{iterations}' + ' '*2
                s += f'val_nll: {val_nll:3.3f}' + ' '*2
                s += f'val_acc : {val_acc_score:2.1%}' + ' '*6
                s += f'Accept rate: {accept_rate:2.1%}' + ' '*2
                s += f'samples per iter: {np.mean(samples_consumed):.0f}'
                s += ' '*(max(0, 100-len(s)))
                print(s[:100], end='')
            # record best model
            if val_nll < best_nll:
                best_nll = val_nll
                best_model = model.copy()
            # check time limit, if given
            if time_limit and time.time() - t0 > time_limit:
                break # if time limit has been surpassed, break
            # reset timer after validation, so validation time isn't counted
            t = time.time()

    # validation best model
    y_pred = best_model(X_val)
    best_val_nll = F.negative_log_likelihood(y_pred, y_val)
    best_val_acc = accuracy_score(y_val, np.argmax(y_pred, axis=1, keepdims=False))
    if verbose:
        # training results
        print('\n')
        print(f'Best model    val_nll: {best_val_nll:.3f}    val_acc: {best_val_acc:.1%}')
        print(f'Acceptance rate: {np.mean(accepted):.1%}')
        t1 = time.time()
        print(f'Wall time: {time.strftime("%M:%S" if t1-t0 < 3600 else "%H:%M:%S", time.gmtime(t1-t0))}    sum of iterations: {np.sum(ms_consumed):.2f} ms    per iteration: {np.mean(ms_consumed):.2f} ms')
        print('#'*21 + ' TRAINING END ' + '#'*21)

    return {
            'val_loss' : val_loss,
            'val_acc' : val_acc,
            'best_val_acc' : best_val_acc,
            'samples_consumed' : samples_consumed,
            'ms_consumed' : ms_consumed,
            'best_model' : best_model,
            'val_freq' : val_freq,
            'label' : label or str(optimizer),
            'test' : getattr(optimizer, 'test', None),
            'batch_size' : batch_size,
            }

# convenience function for running a number of models
def train_wrapper(test, lr, T, batch_size):
    model = initial_model.copy()
    if test in ['naive', 'Barker', 'exact']:
        optimizer = optim.MMH(model, lr=lr, T=T, test=test)
        val_freq = 100
        label = f'{test} b={batch_size}'
    else:
        optimizer = optim.MH(model, lr=lr, T=T)
        val_freq = 20
        label='full batch'

    hist = train(model=model, optimizer=optimizer, Xy_splits=Xy_splits, batch_size=batch_size, iterations=100_000, val_freq=val_freq, time_limit=3, verbose=True, label=label)
    return hist

# Run a selection of models

history = []

runs = [
        {'test' : 'full batch', 'batch_size' : None, 'lr' : 3e-3, 'T' : 7e-4},
        {'test' : 'naive',      'batch_size' : 2,    'lr' : 1e-3, 'T' : 1e-5},
        {'test' : 'naive',      'batch_size' : 64,   'lr' : 1e-3, 'T' : 5e-5},
        {'test' : 'Barker',     'batch_size' : 2,    'lr' : 1e-3, 'T' : 0},
        {'test' : 'Barker',     'batch_size' : 64,   'lr' : 1e-3, 'T' : 0},
        {'test' : 'exact',      'batch_size' : 2,    'lr' : 3e-4, 'T' : 3e-3},
        {'test' : 'exact',      'batch_size' : 64,   'lr' : 1e-3, 'T' : 3e-3},
        ]

for run in runs:
    hist = train_wrapper(**run)
    history.append(hist)

for hist in sorted(history, key=lambda h: -h['best_val_acc']):
    print(hist['label'].ljust(15) + f":  {hist['best_val_acc']:.1%}" + ' accuracy')


# plots

def plot_confusion_matrix(history, ax):
    best_pred = np.argmax(history['best_model'](X_val), axis=1, keepdims=False)
    ConfusionMatrixDisplay.from_predictions(y_val, best_pred, ax=ax, cmap='magma', colorbar=False, im_kw={'vmin':0, 'vmax':180})
    ax.set_title(f"{history['label']}: {history['best_val_acc']:.1%} acc")

def plot_by_samples(metric, history, ax):
    samples = np.cumsum(history['samples_consumed'])
    samples = samples[::history['val_freq']]
    ax.plot(samples, history[metric], label=history['label'])

def plot_by_ms(metric, history, ax):
    ms = np.cumsum(history['ms_consumed'])
    ax.plot(ms, history[metric], label=history['label'])

if False:
    batch_total = samples_consumed
    bins = range(0, np.max(batch_total)+1, batch_size)
    plt.hist(batch_total, bins=np.array(bins)+batch_size/2)
    plt.xticks(bins[1::1+len(bins)//10])
    plt.title('Batch total size')
    plt.ylabel('frequency')
    plt.show()

def plot_batch_hist(history, ax):
    batches = history['samples_consumed']
    batch_size = history['batch_size']
    bins = np.arange(0, np.max(batches)+1, batch_size)
    ax.hist(batches, bins=bins + batch_size/2)
    ax.set_xticks(bins[1::1+len(bins)//10])
    ax.set_title(hist['label'])
    ax.set_xlabel('batch size')
    ax.set_ylabel('frequency')

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

for hist, ax in zip(history, axs.ravel()):
    plot_confusion_matrix(hist, ax)

plt.suptitle('Confusion Matrix')
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))

for hist in history:
    plot_by_samples('val_loss', hist, ax=ax1)
ax1.set_title('Validation loss by data consumption')
ax1.set_xlabel('Number of samples consumed')
ax1.set_ylabel('Validation loss')
ax1.set_xscale('log')
ax1.legend()

for hist in history:
    plot_by_samples('val_acc', hist, ax=ax2)
ax2.set_title('Validation accuracy by data consumption')
ax2.set_xlabel('Number of samples consumed')
ax2.set_ylabel('Validation accuracy')
ax2.set_xscale('log')
ax2.set_ylim(0, 1)
ax2.legend()

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))

for hist in history:
    plot_by_ms('val_loss', hist, ax=ax1)
ax1.set_title('Validation loss by training time')
ax1.set_xlabel('Milliseconds of training time')
ax1.set_ylabel('Validation loss')
ax1.legend()

for hist in history:
    plot_by_ms('val_acc', hist, ax=ax2)
ax2.set_title('Validation accuracy by training time')
ax2.set_xlabel('Milliseconds of training time')
ax2.set_ylabel('Validation accuracy')
ax2.set_ylim(0, 1)
ax2.legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

exact_history = [h for h in history if h['test']=='exact']
for hist, ax in zip(exact_history, axs.ravel()):
    if hist['test'] == 'exact':
        plot_batch_hist(hist, ax)

plt.suptitle('Batch size histogram for exact optimizers')
plt.tight_layout()
plt.show()
