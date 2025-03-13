import itertools as it
import os.path
import random
import sys
from collections import defaultdict
from math import comb, log, prod

import numpy as np
from simulate.kyber import sample_secret_coefs
from simulate.max_likelihood import (
    FalsePositiveNegativePositionalOracle,
    SimpleOracle,
    s_distribution_for_all_y,
)
from simulate_rs import DecoderKyberB2SW2, DecoderKyberB2SW4

ETA = 2


def secret_distribution(eta):
    B = eta
    n = 2 * B
    den = 2**n
    return {s: (comb(n, s + B) / den) for s in range(-B, B + 1)}


def secret_joint_range(bound, weight):
    for s in it.product(range(-bound, bound + 1), repeat=weight):
        # yield s[::-1]
        yield s


def s_joint_to_index(s, bound):
    res = 0
    mult = 1
    for s_val in s:
        res += (s_val + bound) * mult
        mult *= 2 * bound + 1
    return res


def secret_joint_prob(s, s_distr):
    return prod(s_distr[s_val] for s_val in s)


def anticyclic_shift(z):
    return [-z[-1]] + z[:-1]


def evaluate_single_oracle(z_values, X_values, signs):
    """
    z_values: list of coefficients for z.
    X_values: list of thresholds X.
    signs: list of signs (>= or <=) for each condition.
    """
    z_current = z_values
    z_shifts = []
    for _ in range(len(X_values)):
        z_shifts.append(z_current)
        z_current = anticyclic_shift(z_current)

    res = []

    for s in secret_joint_range(ETA, weight=len(z_values)):
        q = False
        for z, X, sign in zip(z_shifts, X_values, signs):
            q = q or ((np.dot(z, s) >= X) if sign == ">=" else (np.dot(z, s) <= X))

        res.append(int(q))

    return res


def coding_for_split(split):
    coding = []

    for single_split in split:
        truth_table = evaluate_single_oracle(
            single_split[0], single_split[1], single_split[2]
        )
        coding.append(truth_table)
    return list(zip(*coding))


# def print_coding(coding, weight=2):
#     header = ["s0", "s1", "\\chi"]
#     print("{:<4} {:<4} {:<15}".format(*header))
#     print("-" * 50)

#     for i, (s0, s1) in enumerate(secret_joint_range(ETA, weight=2)):
#         print("{:<4} {:<4} {:<15}".format(s0, s1, str(coding[i])))


def print_coding(coding, weight=2):
    header = [f"s{i}" for i in range(weight)] + ["\\chi"]
    print(("{:<4} " * weight + "{:<15}").format(*header))
    print("-" * (6 * weight + 15))

    for i, s in enumerate(secret_joint_range(ETA, weight=weight)):
        row_format = "{:<4} " * weight + "{:<15}"
        print(row_format.format(*s, str(coding[i])))


def has_length_4_cycle(matrix):
    # Get the number of check variables
    num_check_vars = len(matrix)

    # Iterate over all pairs of check variables (i, j)
    for i in range(num_check_vars):
        for j in range(i + 1, num_check_vars):
            # Check if there are 2 or more common secret variables
            if np.sum(matrix[i] & matrix[j]) >= 2:
                print(i, j, matrix[i], matrix[j])
                return True  # A length-4 cycle is found

    return False  # No length-4 cycles found


def entropy(distr):
    if type(distr) is dict or type(distr) is defaultdict:
        return -sum(p * log(p, 2) for p in distr.values() if p != 0)
    else:
        return -sum(p * log(p, 2) for p in distr if p != 0)


def marginal_joint(cond_prob_all, weight, s_idx):
    y_len = len(cond_prob_all)
    s_marj_cond = np.zeros((y_len, (2 * ETA + 1)), dtype=np.float32)
    for y in range(y_len):
        for j, s in enumerate(secret_joint_range(ETA, weight=weight)):
            s_val = s[s_idx]
            s_marj_cond[y][s_val + ETA] += cond_prob_all[y][j]
    return s_marj_cond


# print(sample_secret_coefs(256))
s_distr = secret_distribution(ETA)


### BEGIN MUTUAL CHECKS TEST

s_entropy = entropy(s_distr)
print(f"Secret entropy: {s_entropy}")


def compute_information_of_configuration(
    splits, secret_indices, variables, pr_oracle, verbose=False
):
    s_joint = []
    for s in secret_joint_range(ETA, weight=variables):
        s_joint.append(secret_joint_prob(s, s_distr))
    s_joint_entropy = entropy(s_joint)
    if verbose:
        print(f"Secret entropy of joint: {s_joint_entropy}")

    coding_joint = np.zeros(
        ((2 * ETA + 1) ** variables, sum(len(split) for split in splits)), dtype=np.int8
    )
    coding_offset = 0
    for split, check_indices in zip(splits, secret_indices):
        coding_for_check = coding_for_split(split)
        xbits = len(coding_for_check[0])
        coding_for_check_dict = {}
        for s, x in zip(secret_joint_range(ETA, weight=weight), coding_for_check):
            coding_for_check_dict[s] = x

        for i, s in enumerate(secret_joint_range(ETA, weight=variables)):
            s_subset = tuple(s[i] for i in range(len(s)) if i in check_indices)
            for j in range(xbits):
                coding_joint[i][j + coding_offset] = coding_for_check_dict[s_subset][j]
        coding_offset += xbits
    # print_coding(coding_joint, weight=variables)

    cond_prob_all, pr_of_y = s_distribution_for_all_y(
        pr_oracle,
        coding_joint,
        lambda: secret_joint_range(ETA, variables),
        lambda s: secret_joint_prob(s, s_distr),
    )
    # print(cond_prob_all)
    print(f"{pr_of_y=}; entropy is {entropy(pr_of_y)}")
    # print(f"H(Y) - 2 H(p) = {entropy(pr_of_y) - 2 * (1 - bsc_entropy)}")

    expected_info = 0
    for i, y in enumerate(it.product(range(2), repeat=len(coding_joint[0]))):
        if pr_of_y[i] == 0:
            continue
        info = s_joint_entropy - entropy(cond_prob_all[i])
        if verbose:
            print(
                f"Information on {y} for joint is {info}, probability to get y is {pr_of_y[i]}"
            )
        expected_info += info * pr_of_y[i]

    return expected_info


weight = 2

### we look at len(splits) checks, each uses different subsets of secret variables, signified by
### secret_indices list. Each split in splits encodes joint variables as len(split) bits
# splits = [
#     [([1, 2], [-1, 3], ["<=", ">="], 131 / 256, 0.0117187500000000)],
#     # [([1, 2], [-1, 3], ["<=", ">="], 131 / 256, 0.0117187500000000)],
#     [([9, 3], [4, 8], [">=", ">="], 131 / 256, 0.0117187500000000)],
#     # [([1, 2], [-1, 3], ["<=", ">="], 131 / 256, 0.0117187500000000)],
#     # [([1, 2], [-1, 3], ["<=", ">="], 131 / 256, 0.0117187500000000)],
#     [([5, 4], [9, -1], [">=", "<="], 131 / 256, 0.0117187500000000)],
#     [([5, 4], [9, -1], [">=", "<="], 131 / 256, 0.0117187500000000)],
#     [([5, 4], [9, -1], [">=", "<="], 131 / 256, 0.0117187500000000)],
#     [([5, 4], [9, -1], [">=", "<="], 131 / 256, 0.0117187500000000)],
# ]

best_splits = [
    ([-10, 1], [-9, 3], ["<=", ">="], 131 / 256, 0.0117187500000000),
    ([5, -10], [-1, -11], ["<=", "<="], 131 / 256, 0.0117187500000000),
    ([7, -3], [9, 3], [">=", ">="], 131 / 256, 0.0117187500000000),
    ([-10, -5], [-11, 4], ["<=", ">="], 131 / 256, 0.0117187500000000),
    ([-9, -5], [4, 14], [">=", ">="], 131 / 256, 0.0117187500000000),
    ([6, -2], [-6, -3], ["<=", "<="], 131 / 256, 0.0117187500000000),
    ([-6, 1], [5, -5], [">=", "<="], 131 / 256, 0.0117187500000000),
    ([3, 2], [1, 5], [">=", ">="], 131 / 256, 0.0117187500000000),
]

variables = 2
# secret_indices = [(0, 1), (0, 2), (1, 2)]
secret_indices = [(0, 1)]

# variables = 6
# secret_indices = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 5)]

# p = 1
p = 0.9
pr_oracle = SimpleOracle(p)

bsc_entropy = 1 - entropy([p, 1 - p])
print(f"H(p) = {1 - bsc_entropy}")
print(f"upper bound on information: {bsc_entropy} per call; total = {bsc_entropy * 2}")

# splits = [best_splits[0:1]]
# expected_info = compute_information_of_configuration(
#     splits, secret_indices, variables, pr_oracle
# )
# print(f"Expected information is {expected_info}")
# print(131 / 256)
# exit()

best_info = 0
best_indices = []
for indices in it.combinations(range(len(best_splits)), 2):
    print(f"{indices=}")
    # splits = list([best_splits[i]] for i in indices)
    splits = [list(best_splits[i] for i in indices)]
    # splits = [splits[0] * 3]

    # for indices in it.product([(0, 7), (1, 2), (3, 5), (4, 6)], range(len(best_splits))):
    #     splits = [list(best_splits[i] for i in indices[0])] + [[best_splits[indices[1]]]]
    #     print(splits)

    expected_info = compute_information_of_configuration(
        splits, secret_indices, variables, pr_oracle
    )
    # print(f"Expected information is {expected_info}")
    if abs(expected_info - best_info) < 0.000001:
        best_indices.append(indices)
    elif expected_info > best_info:
        best_info = expected_info
        best_indices = [indices]
    break

print("best info:")
print(best_info)
print(best_indices)
exit()


# (([-10, 1], [-9, 3], ['<=', '>='], 0.51171875, 0.01171875), ([-9, -5], [4, 14], ['>=', '>='], 0.51171875, 0.01171875))
# Expected information is 1.9809279232543453
# (([-10, 1], [-9, 3], ['<=', '>='], 0.51171875, 0.01171875), ([-9, -5], [4, 14], ['>=', '>='], 0.51171875, 0.01171875))
# Expected information is 1.8869555777613232

for split in splits:
    coding = coding_for_split(split)
    # print_coding(coding)

    pr_oracle = SimpleOracle(1)
    cond_prob_all = s_distribution_for_all_y(
        pr_oracle,
        coding,
        lambda: secret_joint_range(ETA, weight),
        lambda s: secret_joint_prob(s, s_distr),
    )

    print(cond_prob_all)
exit()


s0_cond = marginal_joint(cond_prob_all, weight, 0)
s1_cond = marginal_joint(cond_prob_all, weight, 1)
print(s0_cond)
print(s1_cond)

print(entropy(s0_cond[0]))
print(entropy(s0_cond[1]))

print(entropy(s1_cond[0]))
print(entropy(s1_cond[1]))

for y in range(len(cond_prob_all)):
    info = 0
    info += s_entropy - entropy(s0_cond[y])
    info += s_entropy - entropy(s1_cond[y])
    print(f"Information on {y} is {info}")

exit()
### END MUTUAL CHECKS TEST

# # 0.001129150390625
# # {(1, 1): 0.234375, (0, 0): 0.25, (0, 1): 0.23828125, (1, 0): 0.27734375}
# split = [
#     ([-3, 1], [-2, 3], ["<=", ">="], 131 / 256, 0.0117187500000000),
#     ([2, -1], [-2, -2], ["<=", "<="], 121 / 256, 0.0273437500000000),
# ]
# weight = 2

# # 5/64 0.078125
# # {'000': 0.140625, '001': 0.09765625, '010': 0.09375, '011': 0.1953125, '100': 0.15625, '101': 0.1328125, '110': 0.13671875, '111': 0.046875}
# weight = 2
# split = [
#     ([4, 1], [-5, -1], ["<=", "<="], 121 / 256, 0.0273437500000000),
#     ([-1, -4], [5, -1], [">=", "<="], 121 / 256, 0.0273437500000000),
#     ([-1, 2], [2, -2], [">=", "<="], 121 / 256, 0.0273437500000000),
# ]

# 0.000213623046875
# {(1, 0): 0.25390625, (1, 1): 0.2578125, (0, 1): 0.25, (0, 0): 0.23828125}
split = [
    (
        [8, 5, -1, 10],
        [-4, 30, -14],
        ["<=", ">=", "<="],
        8175 / 16384,
        0.00103759765625000,
    ),
    (
        [-10, 14, 7, -10],
        [2, -58, 32],
        [">=", "<=", ">="],
        16703 / 32768,
        0.00973510742187500,
    ),
]
weight = 4

# 57/4096 0.013916015625
# {'000': 0.138916015625, '001': 0.13031005859375, '010': 0.1347808837890625, '011': 0.121063232421875, '100': 0.13214111328125, '101': 0.1162872314453125, '110': 0.1126861572265625, '111': 0.1138153076171875}
# split = [
#     ([3, 4, -2, 4], [-1, 14], ["<=", ">="], 31125 / 65536, 0.0250701904296875),
#     ([2, 1, 4, 1], [-2, 6], ["<=", ">="], 31611 / 65536, 0.0176544189453125),
#     ([-2, -2, -2, -3], [-9, 1], ["<=", ">="], 15777 / 32768, 0.0185241699218750),
# ]
# weight = 4


# split = [
#     ([-3, 11, 9, -2], [0, -43], ["<=", "<="], 33557 / 65536, 0.0120391845703125),
#     ([11, -5, -11, 7], [-55, -1], ["<=", "<="], 32011 / 65536, 0.0115509033203125),
#     ([10, -12, 11, -10], [5, -10], [">=", "<="], 1053 / 2048, 0.0141601562500000),
#     ([10, -14, 4, -4], [-6, -23], ["<=", "<="], 4051 / 8192, 0.00549316406250000),
# ]
# weight = 4


coding = coding_for_split(split)
# print_coding(coding)

pr_oracle = SimpleOracle(1)
cond_prob_all = s_distribution_for_all_y(
    pr_oracle,
    coding,
    lambda: secret_joint_range(ETA, weight),
    lambda s: secret_joint_prob(s, s_distr),
)

# print(cond_prob_all)

# fmt: off
# s = [-1, -1, 0, 1]
# s = [-2, -2, -1, 0]
s = [1, 2, 1, -1, 1, 1, 0, 1, -1, 0, -1, 1, 2, 0, -1, 0, 1, 0, -1, 0, 0, 1, -1, 0, -2, -1, 0, 1, 0, -1, 0, -1, -1, 2, -1, -1, 0, 1, -1, -1, -2, -1, -2, -1, 0, 0, 0, -1, 0, -1, -1, 0, -1, 1, -1, 0, 0, -1, 0, 0, 0, 0, 1, 2, 0, 1, 1, 1, 1, 1, 1, -1, 0, -1, 1, 1, -1, 1, 1, -1, 1, 1, 0, -2, -1, 0, 2, 1, -2, 0, 1, 1, 0, 1, -1, 0, -1, 0, 1, 0, 0, 0, -1, 1, 1, 0, 1, 0, 0, 1, 1, 0, -1, -1, 1, 2, 0, 0, -1, -1, -1, -2, -1, -2, 2, -1, 1, 1, -1, 0, 2, -1, 0, -1, 1, -1, 1, 0, 1, -2, 1, 0, -1, 0, 0, 0, -1, 1, -1, 0, -2, 1, 0, 0, 1, -1, 0, 0, 0, 1, 1, 0, 0, 0, 1, -1, 0, -1, 0, 0, -1, -1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, -1, 2, 1, 1, 0, 1, 0, 2, 0, 1, 1, 1, -1, 1, -1, 0, 0, 1, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1, 1, 0, 2, 0, -1, -2, -1, -1, 1, 2, 0, 1, 0, -1, 0, -2, 1, 0, 0, -1, 0, 2, 2, 1, 0, -1, 0, 0, 2, 0, 0, 0, 1, -1, -1, 0, 2, 0, -1, 0, 0, 0, -1]
# fmt: on
print(f"expected = {s}")

s_prior = list(list(s_distr.values()) for _ in range(len(s)))
secret_variables = np.array(s_prior, dtype=np.float32)


random.seed(1)
row = np.array([1] * weight + [0] * (len(s) - weight))
random.shuffle(row)
H = np.array([np.roll(row, shift) for shift in range(256)])
row = np.array([1] * weight + [0] * (len(s) - weight))
random.shuffle(row)
Hprime = np.array([np.roll(row, shift) for shift in range(256)])
H = np.vstack((H, Hprime))

# H = np.array(
#     [
#         [1, 1, 0, 0],
#         [1, 0, 1, 0],
#         [1, 0, 0, 1],
#         [0, 1, 1, 0],
#         # [0, 1, 0, 1],
#         [0, 0, 1, 1],
#     ],
#     dtype=int,
# )
DV = np.max(np.sum(H, axis=0))

# print(H)
print(H.shape)
print(DV)
assert not has_length_4_cycle(H)

check_variables = []
for row in H:
    s_filtered = [s_val for s_val, idx in zip(s, row) if idx == 1]
    check_variables.append(s_filtered)

# print(check_variables)

for i, joint_s in enumerate(check_variables):
    # TODO: actually apply pr_oracle to sample from theoretical value
    y = coding[s_joint_to_index(joint_s, ETA)]
    check_variables[i] = cond_prob_all[int("".join(map(str, y)), 2)]
check_variables = np.array(check_variables, dtype=np.float32)


# append identity matrix
H = np.hstack((H, np.identity(H.shape[0], dtype=int)))


if weight == 2:
    decoder_class = DecoderKyberB2SW2
elif weight == 4:
    decoder_class = DecoderKyberB2SW4
decoder = decoder_class(H.astype("int8"), DV, 15)
s_decoded = decoder.min_sum(secret_variables, check_variables)

print(f"actual = {s_decoded}")

for i, (s_exp, s_act) in enumerate(zip(s, s_decoded)):
    if s_exp != s_act:
        print(f"i: {s_exp} != {s_act}")
