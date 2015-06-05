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


matplotlib.rc('font', size = 20)

def load_data(base_path):
    cores = [a + 1 for a in range(8)]
    strs = []
    #strs.append(('seq-1', ''))
    strs.extend([('mpi-%d' % c, '.%d' % (c + 1)) for c in cores])

    res = []

    path_pattern = '%s.prototxt%s.bin.log'

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

# comp: default for acc kind
def plot_speedup(res_list, xname, yname, pdf, comp = lambda x, y: x >= y, scale = False):
    Y = []
    for core, res in res_list:
        Y.extend([r[yname] for r in res])

    Y.sort()
    mid = Y[len(Y) / 2]

    speedup = []
    names = []
    for core, res in res_list:
        for v in res:
            if comp(v[yname], mid):
                break
        speedup.append(v[xname])
        names.append(core)

    for i in range(1, len(speedup)):
        speedup[i] = speedup[0] / speedup[i]
        if scale:
            speedup[i] *= i

    speedup[0] = 1

    print '\t'.join(['%.2f' % s for s in speedup])

    plt.plot(range(len(speedup)), speedup, marker = '*')
    plt.grid(True)
    plt.xticks(range(len(speedup)), names)
    plt.xlabel('number of machines')
    plt.ylabel('speed up related to %s' % names[0])
    plt.title('%s speed up of %s ~ #machines' % (xname, yname))

    pdf.savefig()
    plt.close()

def plot_sub(res_list, xname, yname, pdf):

    if xname == 'ts(ms)':
        legd = 'ts(1000s)'
        scale = 1000.0 * 1000.0
    else:
        legd = xname
        scale = 1.0


    for core, res in res_list:

        x = [r[xname]/scale for r in res]
        y = [r[yname] for r in res]

        plt.plot(x, y)

    plt.legend([res[0] for res in res_list])
    plt.ylabel(yname)
    plt.xlabel(legd)
    plt.title('%s ~ %s' % (yname, legd))

    pdf.savefig()
    plt.close()

def main():

    import sys
    base_path = sys.argv[1]
    save_path = os.path.basename(base_path) + '-plots.pdf'

    res = load_data(base_path)
    print base_path

    with PdfPages(save_path) as pdf:
        plot_sub(res, 'iter', 'loss', pdf)
        plot_sub(res, 'ts(ms)', 'loss', pdf)
        plot_speedup(res, 'iter', 'loss', pdf, comp = lambda x, y: x <= y, scale = True)
        plot_speedup(res, 'ts(ms)', 'loss', pdf, comp = lambda x, y: x <= y)
        return

        plot_sub(res, 'iter', 'log(|g|^2)', pdf)
        plot_sub(res, 'iter', 'acc', pdf)
        plot_sub(res, 'ts(ms)', 'loss', pdf)
        plot_sub(res, 'ts(ms)', 'log(|g|^2)', pdf)
        plot_sub(res, 'ts(ms)', 'acc', pdf)
        plot_speedup(res, 'ts(ms)', 'acc', pdf)
        plot_speedup(res, 'iter', 'acc', pdf, scale = True)

    pass

if __name__ == "__main__":
    main()

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
