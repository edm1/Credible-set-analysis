#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import os
import pandas as pd
import numpy as np
import argparse

def main():

    # Parse args
    args = parse_arguments()

    # Load
    df = pd.read_csv(args.i, sep="\t")

    # Drop NA
    df = df.loc[~pd.isnull(df[args.col]), :]
    
    # Calc sum of posterior probabilities
    post_sum = reduce(sum_log10, list(df[args.col]))

    # Add posterior probs to table
    df["post_prob"] = df[args.col].apply(lambda logbf: (10**logbf) / (10**post_sum))
    
    # Add cumulative posterior prob
    df = df.sort_values("post_prob", ascending=False)
    df["post_prob_cumsum"] = np.cumsum(df["post_prob"])

    df.to_csv(args.o, sep="\t", index=False)

    return 0

def sum_log10(a, b):
    return np.log10(10**a + 10**b)

def parse_arguments():
    """ Parses command line arguments.
    """
    # Create top level parser.
    parser = argparse.ArgumentParser(description="Calculate credible set.")

    # Add options
    parser.add_argument('--i', metavar="<input>",
        help=('Input file'),
        required=True,
        type=str)
    parser.add_argument('--o', metavar="<output>",
        help=('Output file'),
        required=True,
        type=str)
    parser.add_argument('--col', metavar="<col name>",
        help=('Column containing SNP log10 bayes factors (Default: bayesian_add_log10_bf)'),
        required=False,
        default="bayesian_add_log10_bf",
        type=str)

    # Parse the arguments
    args = parser.parse_args()

    # Parse the arguments
    return args

if __name__ == '__main__':

    main()
