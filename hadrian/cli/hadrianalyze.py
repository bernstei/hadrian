#!/usr/bin/env python3

import numpy as np
from pandas import DataFrame

from .. import Hadrian

def hadrianalyze(args):
    d = np.loadtxt(args.infile)

    h = Hadrian(block_size_factor=args.block_size_factor)
    h.process_traj(d)

    channel_mean, channel_std_err, channel_decor_t = h.means_and_std_errs(plot_file=args.plot_convergence_file)

    d = DataFrame({'mean': channel_mean, 'std err.': channel_std_err, 'decor. time': channel_decor_t})
    d.index.name = 'channel'
    lines = d.to_string().splitlines()
    print("#", lines[0])
    print("#", lines[1])
    print("  " + "\n  ".join(lines[2:]))

def main(*cli_args):
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Analyze a trajectory of samples for their means and "
                                        "*decorrelated* standard errors")
    parser.add_argument("--identical_channels", "-i", action="store_true", help="Assume channels are identical and pool data")
    parser.add_argument("--block_size_factor", "-b", type=float, help="Factor between consecutive block sizes", default=np.sqrt(2))
    parser.add_argument("--min_n_blocks", "-n", type=int, help="minimum number of blocks", default=20)
    parser.add_argument("--plot_convergence_file", "-p", help="file to plot convergence of blocking procedure")
    parser.add_argument("infile", help="input files with sample data, one sample per row, arbitrary number of columns")
    args = parser.parse_args(*cli_args)

    hadrianalyze(args)
