import sys

from simulate.make_code import generate_regular_ldpc_as_tanner

argv = sys.argv
if len(argv) < 5:
    print("Usage: <program> <num_variables> <num_checks> <row_weight> <out_file>")
    exit()

n = int(argv[1])
k = int(argv[2])
row_weight = int(argv[3])
checks = generate_regular_ldpc_as_tanner(n, k, row_weight)
# sort the pairs in a way that checks concentrated in the left part are first
checks = sorted(checks, key=lambda x: x[::-1])

with open(argv[4], "wt") as f:
    for idxs in checks:
        print(",".join(str(idx) for idx in idxs), file=f)
