from typing import Callable, Any

from keras import backend as K

EPS = K.epsilon()  # ~1e-7; prevents /0 and log(0)

def jaccard_coef(y_true, y_pred, smooth=EPS):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)

def dice_coef(y_true, y_pred, smooth=EPS):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2.0 *inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred): return 1. - dice_coef(y_true, y_pred)
def jaccard_coef_loss(y_true, y_pred): return 1. - jaccard_coef(y_true, y_pred)


ALPHA, GAMMA = 0.8, 2.0
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):

    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)

    return focal_loss

################################################################

def bce_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # binary_crossentropy returns per-element; take mean
    return K.mean(K.binary_crossentropy(y_true_f, y_pred_f))

# --- Generic hybrid loss factory ---
def make_hybrid_loss(loss_fns, weights=None, name=None):
    if weights is None:
        weights = [1.0 / len(loss_fns)] * len(loss_fns)
    assert len(weights) == len(loss_fns)

    def _loss(y_true, y_pred):
        return sum(w * fn(y_true, y_pred) for w, fn in zip(weights, loss_fns))

    _loss.__name__ = name or "hybrid_" + "_".join(fn.__name__ for fn in loss_fns)
    return _loss


# --- Common hybrids ---
# BCE + Dice (often a solid default)
bce_dice = make_hybrid_loss([bce_loss, dice_loss], weights=[0.5, 0.5], name="bce_dice")

# BCE + Jaccard
bce_jaccard = make_hybrid_loss([bce_loss, jaccard_coef_loss], weights=[0.5, 0.5], name="bce_jaccard")

# BCE + Focal (helps with class imbalance + hard examples)
bce_focal: Callable[[Any, Any], int] = make_hybrid_loss([bce_loss, FocalLoss], weights=[0.5, 0.5], name="bce_focal")

# Triplet: BCE + Dice + Focal (tweak weights to taste)
bce_dice_focal = make_hybrid_loss([bce_loss, dice_loss, FocalLoss],
                                  weights=[0.4, 0.4, 0.2],
                                  name="bce_dice_focal")