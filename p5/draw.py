import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import argparse


# run on terminal
#
parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

path = args.path
print(path)


def main():
    data_root = 'logs/CartPole-v1'

    fig, ax = plt.subplots(figsize=(8, 5))

    all_ys = []
    # for data_dir in data_dirs:
    record_fname = f'{data_root}/{path}/train_performance/training_record.csv'

    results = pd.read_csv(record_fname, header=None)
    # seed = int(data_dir[-1])
    ys = results[1].values
    all_ys.append(ys)

    xs = list(range(1, len(ys)+1))

    ax.plot(xs, ys, label='SEED = 1', alpha=0.7)

    all_ys = np.array(all_ys)

    ax.plot(xs, all_ys.mean(axis=0), label='mean', linestyle='--')

    ax.legend()
    ax.grid(axis='y')

    ax.set_facecolor('whitesmoke')

    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Fitness', fontsize=16)
    plt.show()


if __name__ == '__main__':
    main()
