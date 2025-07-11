import itertools as it
import random

import numpy as np


class BaseOracle:
    def __init__(self):
        self.oracle_calls = 0

    def prob_of(self, expected, actual, pos):
        raise NotImplementedError()


# Implements binary symmetric channel with probability p to output correct bit
class SimpleOracle(BaseOracle):
    # p: accuracy of oracle
    def __init__(self, p):
        super().__init__()
        self.p = p

    def prob_of(self, expected, actual, pos):
        if actual == expected:
            return self.p
        else:
            return 1 - self.p

    def predict_bit(self, actual_bit, pos):
        self.oracle_calls += 1
        rnd = random.random()
        if rnd < self.p:
            return actual_bit
        else:
            return 1 - actual_bit


# Implements binary channel with different probabilities for output depending on input, i.e.
# there is false positive and false negative probabilities
class FalsePositiveNegativePositionalOracle(BaseOracle):
    # as input p_positional should be array of tuples, where tuple i contains a pair of
    # probability of false positive and false negative, resp., for pos == i.
    # Alternatively, p_positional could be an dictionary, with values being tuples as before
    def __init__(self, p_positional):
        super().__init__()
        self.p_positional = p_positional

    def prob_of(self, expected, actual, pos):
        pr_fp, pr_fn = self.p_positional[pos]
        if expected == 0:
            if actual == 1:
                return pr_fp
            else:
                return 1 - pr_fp
        if expected == 1:
            if actual == 0:
                return pr_fn
            else:
                return 1 - pr_fn

    def predict_bit(self, actual_bit, pos):
        self.oracle_calls += 1
        rnd = random.random()
        pr_fp, pr_fn = self.p_positional[pos]
        if actual_bit == 0:
            if rnd < pr_fp:
                return 1 - actual_bit
            else:
                return actual_bit
        else:
            if rnd < pr_fn:
                return 1 - actual_bit
            else:
                return actual_bit


def pr_cond_yx(y, x, pr_oracle):
    # compute Pr[Y = y | X = x]
    res = 1
    for i in range(len(x)):
        res *= pr_oracle.prob_of(x[i], y[i], i)
    return res


def pr_of_y_from_prediction(pred_y, y):
    res = 1
    for p, yval in zip(pred_y, y):
        if yval == 0:
            res *= 1 - p
        else:
            res *= p
    return res


# Compute Pr[S = s | Y = y] for a given y. Assume that coding have the same number
# of bits for all s
def s_distribution_from_hard_y(y, pr_oracle, coding, s_pmf_array):
    assert coding is not None and len(coding) >= 1 and len(coding[0]) >= 1
    distr = [0] * len(coding)
    for i, (x, pr) in enumerate(zip(coding, s_pmf_array)):
        distr[i] = pr_cond_yx(y, x, pr_oracle) * pr
    pr_y = sum(distr)
    for i in range(len(coding)):
        distr[i] /= pr_y
    return distr

# Compute Pr[S = s | Y = y] for all possible y. Assume that coding have the same number
# of bits for all s
# return Pr[S = s | Y = y] and Pr[Y = y] for all y
def s_distribution_for_all_y(pr_oracle, coding, s_pmf_array):
    assert coding is not None and len(coding) >= 1 and len(coding[0]) >= 1
    ybits = len(coding[0])

    res = np.zeros((2**ybits, len(coding)), dtype=np.float32)
    for j, (x, pr) in enumerate(zip(coding, s_pmf_array)):
        for i, y in enumerate(it.product(range(2), repeat=ybits)):
            res[i][j] = pr * pr_cond_yx(y, x, pr_oracle)

    # Compute Pr[Y = y]
    pr_of_y = np.sum(res, axis=1)

    # Here we divide each probability by Pr[Y = y]
    for i in range(2**ybits):
        pr = pr_of_y[i]
        if pr == 0:
            # initialize all probabilities to nan
            res[i] = None
        else:
            res[i] = res[i] / pr
    return res, pr_of_y


def s_distribution_from_prediction_y(
    pred_y, pr_oracle, secret_range_func, coding, distrib_secret, sum_weight
):
    distr = [0] * len(secret_range_func(sum_weight))
    for y in it.product(range(2), repeat=len(coding[0])):
        pr_y_saved = pr_y(
            y, pr_oracle, secret_range_func, coding, distrib_secret, sum_weight
        )
        for i, s in enumerate(secret_range_func(sum_weight)):
            distr[i] += pr_cond_xy(
                s,
                y,
                pr_oracle,
                secret_range_func,
                coding,
                distrib_secret,
                sum_weight,
                pr_y_saved,
            ) * pr_of_y_from_prediction(pred_y, y)
    return distr


def pr_cond_yx_adaptive(y, s, pr_oracle, coding_tree):
    # compute Pr[Y = y | X = e(s)]
    res = 1
    node = coding_tree
    for y_val in y:
        pos = (node.ge_flag, node.value)
        if node.ge_flag:
            expected_bit = int(s >= node.value)
        else:
            expected_bit = int(s <= node.value)
        res *= pr_oracle.prob_of(expected_bit, y_val, pos)
        if y_val == 1:
            node = node.right
        else:
            node = node.left
    return res


def pr_y_adaptive(
    y, pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
):
    # compute Pr[Y = y]
    res = 0
    for s in secret_range_func(sum_weight):
        pr_xprime_y = distrib_secret[s] * pr_cond_yx_adaptive(
            y, s, pr_oracle, coding_tree
        )
        res += pr_xprime_y
    return res


def pr_cond_xy_adaptive(
    s,
    y,
    pr_oracle,
    secret_range_func,
    coding_tree,
    distrib_secret,
    sum_weight,
    pr_y_saved=None,
):
    # compute Pr[X = e(s) | Y = y]
    if pr_y_saved is None:
        pr_y_saved = pr_y_adaptive(
            y, pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
        )
    return (
        pr_cond_yx_adaptive(y, s, pr_oracle, coding_tree)
        * distrib_secret[s]
        / pr_y_saved
    )


# Return the conditional probability given some output y that was obtained
# following a tree. It is computed by trying all possible inputs of the secret
def s_distribution_from_hard_y_adaptive(
    y, pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
):
    # assume here that distrib_secret include probabilities for all values, i.e.
    # distrib_secret[s] is 0 for non-existent s in original distrib_secret
    distr = [0] * (2 * sum_weight + 1)
    pr_y_saved = pr_y_adaptive(
        y, pr_oracle, secret_range_func, coding_tree, distrib_secret, sum_weight
    )
    for i, s in enumerate(secret_range_func(sum_weight)):
        distr[i] = pr_cond_xy_adaptive(
            s,
            y,
            pr_oracle,
            secret_range_func,
            coding_tree,
            distrib_secret,
            sum_weight,
            pr_y_saved,
        )
    return distr


# As an input we have a prediction y, i.e. for some secret value we follow the tree, then some oracle not only choosing left or right, but giving certainty that we should go left or right (it is treated as probability), we go left if this probability < 0.5, right otherwise. In the adaptive case we can't check the all tree, we essentially have hard y (the checks are fixed), we just modify conditional probability by certainty of prediction y
def s_distribution_from_prediction_y_adaptive(
    pred_y, secret_range_func, coding_tree, distrib_secret, sum_weight
):
    hard_y = tuple(round(pred_y_val) for pred_y_val in pred_y)
    # assume here that distrib_secret include probabilities for all values, i.e.
    # distrib_secret[s] is 0 for non-existent s in original distrib_secret
    distr = [0] * (2 * sum_weight + 1)
    for i, s in enumerate(secret_range_func(sum_weight)):
        node = coding_tree
        pr = distrib_secret[s]
        for y_val, y_pred_val in zip(hard_y, pred_y):
            if node.ge_flag:
                expected_bit = int(s >= node.value)
            else:
                expected_bit = int(s <= node.value)
            if expected_bit == 0:
                pr *= 1 - y_pred_val
            else:
                pr *= y_pred_val
            if y_val == 1:
                node = node.right
            else:
                node = node.left
        distr[i] = pr
    # normalizing step
    t = sum(distr)
    for i in range(len(distr)):
        distr[i] /= t
    return distr