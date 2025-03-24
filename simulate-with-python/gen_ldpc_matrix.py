import os
import sys

library_path = os.path.join(os.path.dirname(__file__), "ProtographLDPC")

if not os.path.exists(library_path):
    print(
        f"ProtographLDPC library is not installed, from {os.path.dirname(__file__)} folder run:"
    )
    print(
        "git submodule init; git submodule update; cd ProtographLDPC; git pull --recurse-submodules; git submodule update --init --recursive; cd LDPC-codes/; make; cd ..; cd peg/; make; cd ../.."
    )
    exit()

sys.path.append(os.path.join(library_path, "LDPC-library"))
from libs.RegularLDPC import RegularLDPC


# generate tall matrix with n variables, k check nodes with fixed row_weight
def generate_regular_ldpc_as_tanner(n, k, row_weight):
    # library can generate transposed matrix
    ldpc_code = RegularLDPC([k, n, row_weight], "peg", verbose=False)
    transposed_tanner = ldpc_code.tanner_graph
    tanner = [[] for _ in range(k)]
    for col_idx, row_idxs in transposed_tanner.items():
        for row_idx in row_idxs:
            tanner[row_idx].append(col_idx)
    return tanner


argv = sys.argv
if len(argv) < 5:
    print("Usage: <program> <num_variables> <num_checks> <row_weight> <out_file>")
    exit()

n = int(argv[1])
k = int(argv[2])
row_weight = int(argv[3])
checks = generate_regular_ldpc_as_tanner(n, k, row_weight)

with open(argv[4], "wt") as f:
    for idxs in checks:
        print(",".join(str(idx) for idx in idxs), file=f)
