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
import os

def load_data(base_path):
    cores = [a + 1 for a in range(7)]
    strs = []
    strs.append(('seq-1', ''))
    strs.extend([('mpi-%d' % c, '.%d' % (c + 1)) for c in cores])

    res = []

    path_pattern = '%s.prototxt%s.log'

    for name, prefix in strs:
        res.append((name, load_data_sub(path_pattern % (base_path, prefix))))

    return res

def load_data_sub(path):

    res = []

    with open(path) as reader:
        for line in reader:
            fields = line.strip().split(' ')
            ITER = {}
            for i in range(0, len(fields), 2):
                ITER[fields[i][:-1]] = float(fields[i+1])
            res.append(ITER)

    return res

def plot_sub(res_list, xname, yname, pdf):


    for core, res in res_list:

        x = [r[xname] for r in res]
        y = [r[yname] for r in res]

        plt.plot(x, y)

    plt.legend([res[0] for res in res_list])
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.title('%s ~ %s' % (yname, xname))

    pdf.savefig()
    plt.close()

def main():

    import sys
    base_path = sys.argv[1]
    save_path = os.path.basename(base_path) + '-plots.pdf'

    res = load_data(base_path)

    with PdfPages(save_path) as pdf:
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
