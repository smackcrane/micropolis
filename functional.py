import numpy as np

def softmax(logits):
    """Softmax: compute probabilities from logits

    logits: (n_samples, n_labels)
    returns probabilities: (n_samples, n_labels)
    """

    logits = logits - np.max(logits, axis=1, keepdims=True) # normalize for numerical stability
    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs

def negative_log_likelihood(y_pred, y, reduction='mean'):
    """Compute negative log likelihood, aka cross-entropy

    y_pred: (n_samples, n_labels) predicted logits
    y: (n_samples,) ground truth labels
    returns float, average negative log likelihood
    """

    probs = softmax(y_pred)
    likelihood = probs[np.arange(len(y)), y]
    if   reduction == 'mean' : return -np.mean(np.log(likelihood))
    elif reduction == 'sum'  : return -np.sum(np.log(likelihood))
    elif reduction == 'none' : return -np.log(likelihood)


