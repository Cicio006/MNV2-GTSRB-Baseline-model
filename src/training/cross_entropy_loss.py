import cupy as cp


def cross_entropy_with_logits(logits: cp.ndarray, labels: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Computes mean softmax cross entropy loss and gradient w.r.t. logits.

    logits: shape (N, C)
    labels: shape (N,), integer class labels
    returns: (loss, grad_logits)
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")

    if labels.ndim != 1:
        raise ValueError("labels must have shape (N,)")

    if logits.shape[0] != labels.shape[0]:
        raise ValueError("batch size mismatch between logits and labels")
    
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("labels contain invalid class indices")

    batch_size = logits.shape[0]

    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = cp.exp(shifted_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    correct_class_probs = probs[cp.arange(batch_size), labels]
    loss = -cp.log(correct_class_probs + 1e-12).mean()

    grad_logits = probs.copy()
    grad_logits[cp.arange(batch_size), labels] -= 1.0
    grad_logits /= batch_size

    return loss, grad_logits