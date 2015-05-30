#!/usr/bin/env python

"""
Python source code - replace this with a description of the code and write the code below this text.
"""

import math, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from collections import defaultdict

def load_data(path):

    res = []

    with open(path) as reader:
        for line in reader:
            fields = line.strip().split(' ')
            ITER = {}
            for i in range(0, len(fields), 2):
                ITER[fields[i][:-1]] = float(fields[i+1])
            res.append(ITER)

    return res

def plot_sub(res, xname, yname, pdf):

    x = [r[xname] for r in res]
    y = [r[yname] for r in res]

    plt.plot(x, y)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.title('%s ~ %s' % (yname, xname))

    pdf.savefig()
    plt.close()

def main():

    import sys

    log_path = sys.argv[1]
    res = load_data(log_path)

    with PdfPages(log_path + '-plots.pdf') as pdf:
        plot_sub(res, 'iter', 'loss', pdf)
        plot_sub(res, 'iter', 'log(|g|^2)', pdf)
        plot_sub(res, 'iter', 'acc', pdf)
        plot_sub(res, 'ts(ms)', 'loss', pdf)
        plot_sub(res, 'ts(ms)', 'log(|g|^2)', pdf)
        plot_sub(res, 'ts(ms)', 'acc', pdf)

    pass

if __name__ == "__main__":
    main()

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
