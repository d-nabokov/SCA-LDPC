import inspect
import logging
import os.path
import re
import sys

import coloredlogs
import numpy as np
from simulate_rs import (
    DecoderExtendedNTRUW2,
    DecoderExtendedNTRUW4,
    DecoderExtendedNTRUW6,
    DecoderNTRUW2,
    DecoderNTRUW4,
    DecoderNTRUW6,
)

logger = logging.getLogger(__name__.replace("__", ""))
coloredlogs.install(level="INFO", logger=logger)

MOVE_SINGLE_CHECKS_TO_APRIOR = True
USE_EXTENDED_VARIABLES = True

if USE_EXTENDED_VARIABLES:
    B = 2
else:
    B = 1


def extended_variables_indices(indices):
    n = len(indices)
    out = []
    i = 0

    while i < n:
        # discover the length of the current consecutive (+1 mod p) run
        run_len = 1
        while (
            i + run_len < n
            and indices[i + run_len] == (indices[i + run_len - 1] + 1) % p
        ):
            run_len += 1

        if run_len == 1:  # single, no pair
            out.append(indices[i])
        elif run_len == 2:  # clean (x, x+1) pair → keep only x+1
            out.append(indices[i + 1])
        else:  # run_len ≥ 3  → ambiguous
            raise ValueError(
                f"Ambiguous input: overlapping (x, x+1) pairs starting at "
                f"position {i} ({indices[i]}, …)"
            )

        i += run_len  # step over the whole run
    return out


def resize_pmf(pmf, target_b):
    target_size = 2 * target_b + 1
    if len(pmf) > target_size:
        offset = (len(pmf) - target_size) // 2
        return pmf[offset:-offset]
    elif len(pmf) < target_size:
        offset = (target_size - len(pmf)) // 2
        return [0.0] * offset + pmf + [0.0] * offset
    else:
        return pmf


def process_cond_prob_file(filename, n, check_weight):
    if not os.path.isfile(filename):
        print("File does not exist")
        return None, None

    with open(filename, "r") as file:
        lines = file.readlines()

    index_lines = []
    probability_lists = []

    single_check_idxs = []
    single_check_distr = []

    col_idx = None
    # pred_col_idx = None
    # last_single_index = None

    # read lines in blocks of 2
    for i in range(0, len(lines), 2):
        indices = list(map(int, lines[i].strip().split(",")))
        probabilities = list(map(float, lines[i + 1].strip().split(",")))

        assert len(list(x for x in probabilities if x != 0)) == len(indices) * 2 + 1

        if USE_EXTENDED_VARIABLES:
            indices = extended_variables_indices(indices)

        # support the case where extra probabilities are not printed
        if len(probabilities) == len(indices) * 2 + 1 and len(indices) < check_weight:
            offset = check_weight - len(indices)
            probabilities = [0.0] * offset + probabilities + [0.0] * offset

        if i == 0:
            col_idx = indices[0]

        # if pred_col_idx is None and len(indices) == 2:
        #     pred_col_idx = col_idx - last_single_index + 1
        # last_single_index = indices[0]

        if MOVE_SINGLE_CHECKS_TO_APRIOR and len(indices) == 1:
            single_check_idxs.append(indices[0])
            single_check_distr.append(probabilities)
        else:
            index_lines.append(indices)
            probability_lists.append(probabilities)

    # Determine the number of parity checks
    num_rows = len(index_lines)
    # Create the matrix with the appropriate size
    matrix = np.zeros((num_rows, n + num_rows), dtype=int)

    # Fill in the ones based on indices and append the negative identity matrix
    for i, indices in enumerate(index_lines):
        for index in indices:
            matrix[i, index] = 1
        matrix[i, n + i] = -1

    return (
        matrix,
        index_lines,
        probability_lists,
        single_check_idxs,
        single_check_distr,
        col_idx,
    )


def parse_key_info_file(file_path):
    keys = []
    collisions = []

    p = re.compile("pq_counter: (\d+),inner_test: (\d+)")

    with open(file_path, "r") as f:
        current_key = []
        in_key_section = False
        collision_info = []

        current_counter = None
        for line in f:
            line = line.strip()

            # Look for "pq_counter:"
            if line.startswith("pq_counter:"):
                m = p.match(line)
                pq_counter = int(m[1])
                # If the counter is different, we finished handling and we save it
                if current_counter is None:
                    current_counter = pq_counter
                elif pq_counter != current_counter:
                    current_counter = pq_counter
                    keys.append(current_key)
                    collisions.append(collision_info)
                # Reset variables for new entry
                current_key = []
                in_key_section = False
                collision_info = []

            # Look for "The private key is:"
            elif line == "The private key is:":
                in_key_section = True
                continue  # Skip to next line, where the key starts

            # If we are in the key section, capture the key data
            elif in_key_section:
                if line:  # If the line contains key data
                    # Remove trailing comma and split the key values
                    current_key = [int(x) for x in line.rstrip(",").split(",")]
                    in_key_section = False  # We are done with key section

            # Capture collision index and value
            elif line.startswith("collision_index"):
                index_value = line.split(",")
                collision_index = int(index_value[0].split(":")[1])
                collision_value = int(index_value[1].split(":")[1])
                collision_info.append((collision_index, collision_value))

    # Don't forget to add the last data
    keys.append(current_key)
    collisions.append(collision_info)
    return keys, collisions


def is_unreliable(pmf, threshold=0.8):
    max_pr = np.max(pmf)
    return max_pr < threshold


def set_unreliable_to_second_most_probable(pmf, tau=0.01):
    sorted_indices = np.argsort(pmf)[::-1]
    second_largest_index = sorted_indices[1]

    new_pmf = np.full_like(pmf, fill_value=tau, dtype=float)

    n = len(pmf)
    new_pmf[second_largest_index] = 1.0 - tau * (n - 1)

    return new_pmf.tolist()


def list_of_unsatisfied_checks(
    f, variable_in_check_idxs, check_variables, col_idx, pred_col_idx
):
    BSUM = len(check_variables[0]) // 2
    unsatisfied_checks = []
    for variable_idxs, check_pmf in zip(variable_in_check_idxs, check_variables):
        is_relevant_part = True
        for variable_idx in variable_idxs:
            is_relevant_part = is_relevant_part and is_from_maj_voting_part(
                variable_idx, col_idx, pred_col_idx
            )
        if not is_relevant_part:
            continue
        beta_u = 0
        for idx in variable_idxs:
            beta_u += f[idx]
        beta_from_pmf = np.argmax(check_pmf) - BSUM
        if beta_u != beta_from_pmf:
            unsatisfied_checks.append(tuple(variable_idxs))
    return unsatisfied_checks


def filter_unreliable_checks(unsatisfied_checks, s_pmfs):
    unreliable_checks = []
    for variable_idxs in unsatisfied_checks:
        for idx in variable_idxs:
            if is_unreliable(s_pmfs[idx]):
                unreliable_checks.append(variable_idxs)
                break
    return unreliable_checks


def find_unreliable_block(s_pmfs, unreliable_idx):
    l = len(s_pmfs)
    idx_lower = unreliable_idx
    idx_upper = unreliable_idx + 1
    while idx_lower > 0 and is_unreliable(s_pmfs[idx_lower - 1]):
        idx_lower -= 1
    while is_unreliable(s_pmfs[idx_upper % l]):
        idx_upper += 1
    return idx_lower, idx_upper


def decode_with_post_block_flip_optimization(
    decoder,
    secret_variables,
    check_variables,
    variable_in_check_idxs,
    col_idx,
    pred_col_idx,
):
    s_decoded_pmfs_orig = decoder.decode_with_pr(secret_variables, check_variables)
    ret = s_decoded_pmfs_orig
    fprime = list(np.argmax(pmf) - 1 for pmf in s_decoded_pmfs_orig)

    # potentially_incorrect = []

    unsatisfied_checks_orig = list_of_unsatisfied_checks(
        fprime, variable_in_check_idxs, check_variables, col_idx, pred_col_idx
    )
    # print(f"{unsatisfied_checks_orig=}")
    cur_unsatisfied_checks = unsatisfied_checks_orig
    cur_s_decoded_pmfs = s_decoded_pmfs_orig
    for i, variable_idxs in enumerate(unsatisfied_checks_orig):
        unreliable_idx = None
        for idx in variable_idxs:
            if is_unreliable(s_decoded_pmfs_orig[idx]):
                # if is_unreliable(cur_s_decoded_pmfs[idx]):
                unreliable_idx = idx
                break
        if unreliable_idx is None:
            continue
        # # TODO: blocks can be the same, keep track of what we tried
        # idx_lower, idx_upper = find_unreliable_block(
        #     s_decoded_pmfs_orig, unreliable_idx
        # )

        # two neighboring checks often define boundaries of incorrect block
        if (
            i < len(unsatisfied_checks_orig) - 1
            and (variable_idxs[0] - unsatisfied_checks_orig[i + 1][1]) < 13
        ):
            idx_lower = unsatisfied_checks_orig[i + 1][1]
            idx_upper = variable_idxs[1]
        else:
            idx_lower, idx_upper = find_unreliable_block(
                cur_s_decoded_pmfs, unreliable_idx
            )

        # print(f"looking at flipping {(idx_lower, idx_upper)}")
        new_secret_variables = secret_variables.copy()
        for idx in range(idx_lower, idx_upper):
            idx = idx % len(s_decoded_pmfs_orig)
            # TODO: change s_decoded_pmfs_orig in the process?
            new_secret_variables[idx] = set_unreliable_to_second_most_probable(
                s_decoded_pmfs_orig[idx],
                tau=0.01,
            )
            # new_secret_variables[idx] = set_unreliable_to_second_most_probable(
            #     cur_s_decoded_pmfs[idx]
            # )
        s_decoded_pmfs = decoder.decode_with_pr(new_secret_variables, check_variables)
        fprime = list(np.argmax(pmf) - 1 for pmf in s_decoded_pmfs)

        unsatisfied_checks = list_of_unsatisfied_checks(
            fprime, variable_in_check_idxs, check_variables, col_idx, pred_col_idx
        )
        # print(f"{unsatisfied_checks=}")
        if len(unsatisfied_checks) < len(cur_unsatisfied_checks):
            # print(
            #     f"managed to reduce number of unsatisfied checks by flipping values in ({idx_lower}, {idx_upper})"
            # )
            cur_unsatisfied_checks = unsatisfied_checks
            secret_variables = new_secret_variables
            ret = s_decoded_pmfs
            cur_s_decoded_pmfs = s_decoded_pmfs
            continue
        # unsatisfied_checks_diff = set(cur_unsatisfied_checks) - set(unsatisfied_checks)
        # new_unsatisfied = set(unsatisfied_checks) - set(cur_unsatisfied_checks)
        # print(f"{cur_unsatisfied_checks=}")
        # print(f"{unsatisfied_checks_diff=}")
        # print(f"{new_unsatisfied=}")
        # if len(new_unsatisfied) == 1:
        #     new_unsatisfied_check = new_unsatisfied.pop()
        #     if (
        #         idx_lower <= new_unsatisfied_check[0] < idx_upper
        #         or idx_lower <= new_unsatisfied_check[1] < idx_upper
        #     ):
        #         idx_lower_new = max(idx_lower, new_unsatisfied_check[0])
        #         idx_upper_new = min(idx_upper, new_unsatisfied_check[1])
        #         print(f"Trying flipping new {(idx_lower_new, idx_upper_new)}")
        #         new_secret_variables = secret_variables.copy()
        #         for idx in range(idx_lower_new, idx_upper_new):
        #             idx = idx % len(s_decoded_pmfs_orig)
        #             new_secret_variables[idx] = set_unreliable_to_second_most_probable(
        #                 s_decoded_pmfs_orig[idx],
        #                 tau=0.01,
        #             )
        #         s_decoded_pmfs_new = decoder.decode_with_pr(
        #             new_secret_variables, check_variables
        #         )
        #         fprime_new = list(np.argmax(pmf) - 1 for pmf in s_decoded_pmfs_new)

        #         unsatisfied_checks_new = list_of_unsatisfied_checks(
        #             fprime_new, variable_in_check_idxs, check_variables, col_idx, pred_col_idx
        #         )
        #         print("AAAAAAAAAAAAAA")
        #         print(f"{unsatisfied_checks_new=}")
        #     new_unsatisfied.add(new_unsatisfied_check)

        # if len(unsatisfied_checks_diff) == 1:
        #     if (idx_lower, idx_upper) not in unsatisfied_checks_diff:
        #         print(
        #             "PROBLEM: if (idx_lower, idx_upper) not in unsatisfied_checks_diff:"
        #         )
        #         print(f"{unsatisfied_checks_diff=}")
        #         print(f"{cur_unsatisfied_checks=}")
        #         print(f"{unsatisfied_checks=}")

        #         continue

        #     if len(new_unsatisfied) != 1:
        #         print("PROBLEM: if len(new_unsatisfied) != 1:")
        #         print(f"{new_unsatisfied=}")
        #         print(f"{cur_unsatisfied_checks=}")
        #         print(f"{unsatisfied_checks=}")

        #         exit()
        #     potential_error = set((idx_lower, idx_upper)).intersection(
        #         new_unsatisfied.pop()
        #     )
        #     if len(potential_error) == 1:
        #         potentially_incorrect.append(potential_error.pop())
    return ret


def is_from_maj_voting_part(i, col_idx, pred_col_idx):
    return not ((col_idx - pred_col_idx + 1) <= i <= col_idx)


# argv = sys.argv
# if len(argv) < 3:
#     print("Usage: <program> <prob_file> <out_file> [<LDPC_iterations>]")
#     exit()

base_data_folder = "conditional probs"
prob_filename = os.path.join(base_data_folder, "private_key_and_collision_info.bin")
outfile = open("outfile.txt", "wt")
filename_pattern = os.path.join(
    base_data_folder, "For NO_TESTS is {} alpha_u_and_conditional_probabilities.bin"
)

keys, collisions = parse_key_info_file(prob_filename)

keys_to_test = range(0, 30)

iterations = 10000
# if len(argv) >= 4:
#     iterations = int(argv[3])
# number of coefficients of f
p = 761
# weight of f
w = 286
check_weight = 4

# determine the prior distribution of coefficients of f
f_zero_prob = (p - w) / p
f_one_prob = (1 - f_zero_prob) / 2

differences_arr = []
maj_voting_part_errors_arr = []
non_maj_voting_part_errors_arr = []
full_recovered_keys = 0
for key_idx in keys_to_test:
    # print(f"working over key {key_idx} with collisions {collisions[key_idx]}")
    if len(collisions[key_idx]) > 1:
        print(f"skipping multiple collision case for {key_idx}")
        continue
    # read posterior distribution of check variables
    filename = filename_pattern.format(key_idx)
    (
        H,
        variable_in_check_idxs,
        check_variables,
        single_check_idxs,
        single_check_distr,
        col_idx,
    ) = process_cond_prob_file(filename, p, check_weight)

    if H is None or check_variables is None:
        exit()
    row_counts = np.count_nonzero(H, axis=1)
    max_row_weight = np.max(row_counts)
    col_counts = np.count_nonzero(H, axis=0)
    max_col_weight = np.max(col_counts)

    if (max_row_weight - 1) > check_weight:
        print(f"skipping too large predicted collision index for {key_idx}")
        continue

    secret_variables = []

    single_checks = sorted(zip(single_check_idxs, single_check_distr))
    single_checks_idx = 0
    for i in range(p):
        if (
            single_checks_idx < len(single_checks)
            and single_checks[single_checks_idx][0] == i
        ):
            distr = single_checks[single_checks_idx][1]
            secret_variables.append(resize_pmf(distr, B))
            single_checks_idx += 1
        else:
            secret_variables.append(
                resize_pmf([f_one_prob, f_zero_prob, f_one_prob], B)
            )

    # convert to numpy arrays for Rust be able to work on the arrays
    secret_variables = np.array(secret_variables, dtype=np.float32)
    check_variables = np.array(check_variables, dtype=np.float32)
    # if collision value is 1, we need to multiply the result by -1
    if collisions[key_idx][0][1] == 1:
        secret_variables = secret_variables[:, ::-1]
        check_variables = check_variables[:, ::-1]

    # Rust does not accept zero values, set them to very small probability
    epsilon = 1e-20
    secret_variables[secret_variables == 0] = epsilon
    check_variables[check_variables == 0] = epsilon

    decoder_map = {
        (False, 2): DecoderNTRUW2,
        (False, 4): DecoderNTRUW4,
        (False, 6): DecoderNTRUW6,
        (True, 2): DecoderExtendedNTRUW2,
        (True, 4): DecoderExtendedNTRUW4,
        (True, 6): DecoderExtendedNTRUW6,
    }
    if USE_EXTENDED_VARIABLES:
        ldpc_check_weight = check_weight // 2
    else:
        ldpc_check_weight = check_weight
    if (USE_EXTENDED_VARIABLES, ldpc_check_weight) not in decoder_map:
        raise ValueError("Not supported check weight")
    decoder = decoder_map[(USE_EXTENDED_VARIABLES, ldpc_check_weight)](
        H.astype("int8"), max_col_weight, max_row_weight, iterations
    )

    # s_decoded_pmfs = decoder.decode_with_pr(secret_variables, check_variables)
    # s_decoded_pmfs = decode_with_post_block_flip_optimization(
    #     decoder, secret_variables, check_variables, variable_in_check_idxs, col_idx, pred_col_idx
    # )
    # for i, pmf in enumerate(s_decoded_pmfs):
    #     print(f"{i}: {pmf}")

    f = keys[key_idx]
    # fprime = list(np.argmax(pmf) - 1 for pmf in s_decoded_pmfs)

    # BSUM = len(check_variables[0]) // 2
    # unsatisfied_checks = 0
    # print("Getting wrong checks even with majority voting:")
    # for variable_idxs, check_pmf in zip(variable_in_check_idxs, check_variables):
    #     if variable_idxs[0] < col_idx:
    #         continue
    #     true_beta = 0
    #     for idx in variable_idxs:
    #         true_beta += f[idx]
    #     beta_from_pmf = np.argmax(check_pmf) - BSUM
    #     if true_beta != beta_from_pmf:
    #         print(f"{variable_idxs}: {true_beta} != {beta_from_pmf}", file=outfile)
    # print("++++++++++")
    # for variable_idxs, check_pmf in zip(variable_in_check_idxs, check_variables):
    #     if variable_idxs[0] < col_idx:
    #         continue
    #     beta_u = 0
    #     true_beta = 0
    #     for idx in variable_idxs:
    #         beta_u += fprime[idx]
    #         true_beta += f[idx]
    #     beta_from_pmf = np.argmax(check_pmf) - BSUM
    #     if beta_u != beta_from_pmf:
    #         unsatisfied_checks += 1
    #         print(
    #             f"{variable_idxs}: {beta_u} != {beta_from_pmf} =? {true_beta}; {'(check pmf with error)' if beta_from_pmf != true_beta else ''}"
    #         )
    #         for idx in variable_idxs:
    #             if is_unreliable(s_decoded_pmfs[idx]):
    #                 print(f"{idx} is unreliable")
    # print(f"{unsatisfied_checks=}")

    # s_decoded_pmfs = decode_with_post_block_flip_optimization(
    #     decoder,
    #     secret_variables,
    #     check_variables,
    #     variable_in_check_idxs,
    #     col_idx,
    #     pred_col_idx,
    # )
    s_decoded_pmfs = decoder.decode_with_pr(secret_variables, check_variables)
    # super_unreliable = list(
    #     idx
    #     for idx in range(col_idx, len(s_decoded_pmfs))
    #     if is_unreliable(s_decoded_pmfs[idx], 0.6)
    # )
    # print(f"very unreliable coefs: {super_unreliable}")
    fprime = list(np.argmax(pmf) - B for pmf in s_decoded_pmfs)
    # differences = sum(f[i] != fprime[i] for i in range(len(f)))
    # if differences == 0:
    #     full_recovered_keys += 1
    # differences_arr.append(differences)

    print(f"{col_idx=}")
    for i in range(p):
        if i > 0 and i <= col_idx:
            expect = f[i]
        else:
            expect = f[i] + f[(i - 1) % p]
        if expect != fprime[i]:
            print(f"{i}: expected {expect}, got {fprime[i]}, pmf = {s_decoded_pmfs[i]}")

    # getting from extended representation back to normal
    num_extended = p - col_idx
    matrix = np.zeros((num_extended, p + num_extended), dtype=int)
    for row_idx, i in enumerate(range(col_idx + 1, p + 1)):
        matrix[row_idx, i % p] = 1
        matrix[row_idx, (i - 1) % p] = 1
        matrix[row_idx, p + row_idx] = -1
    row_counts = np.count_nonzero(matrix, axis=1)
    max_row_weight = np.max(row_counts)
    col_counts = np.count_nonzero(matrix, axis=0)
    max_col_weight = np.max(col_counts)

    secret_variables = []
    for i in range(p):
        if i > 0 and i <= col_idx:
            pmf = s_decoded_pmfs[i]
            secret_variables.append(resize_pmf(pmf, 1))
        else:
            secret_variables.append(
                resize_pmf([f_one_prob, f_zero_prob, f_one_prob], 1)
            )
    secret_variables = np.array(secret_variables, dtype=np.float32)
    check_variables = s_decoded_pmfs[col_idx + 1 :] + s_decoded_pmfs[0:1]
    check_variables = np.array(check_variables, dtype=np.float32)
    decoder = decoder_map[(False, 2)](
        matrix.astype("int8"), max_col_weight, max_row_weight, iterations
    )
    s_decoded_pmfs = decoder.decode_with_pr(secret_variables, check_variables)
    print("Switching to non-extended representation")
    fprime = list(np.argmax(pmf) - 1 for pmf in s_decoded_pmfs)
    differences = 0
    for i in range(p):
        if f[i] != fprime[i]:
            differences += 1
            print(f"{i}: expected {f[i]}, got {fprime[i]}, pmf = {s_decoded_pmfs[i]}")
    differences_arr.append(differences)

    print(f"For key {key_idx} have total {differences} errors:", file=outfile)
    # maj_voting_part_errors = 0
    # non_maj_voting_part_errors = 0
    # for i in range(len(f)):
    #     if f[i] != fprime[i]:
    #         if is_from_maj_voting_part(i, col_idx, pred_col_idx):
    #             maj_voting_part_errors += 1
    #             ending = "from majority"
    #         else:
    #             non_maj_voting_part_errors += 1
    #             ending = "from paired"
    #         print(f"pos {i}: expected {f[i]}, got {fprime[i]};  {ending}", file=outfile)
    # maj_voting_part_errors_arr.append(maj_voting_part_errors)
    # non_maj_voting_part_errors_arr.append(non_maj_voting_part_errors)
    # print(f"{potentially_incorrect=}", file=outfile)
    # print("\n=============================\n")

outfile.close()

print(f"Managed to fully recover {full_recovered_keys} keys")
# print(
#     f"Average number of errors from majority voting part is {np.average(maj_voting_part_errors_arr)}"
# )
# print(
#     f"Average number of errors from non majority voting part is {np.average(non_maj_voting_part_errors_arr)}"
# )
print(f"Average number of errors total is {np.average(differences_arr)}")