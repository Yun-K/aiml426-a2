import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():

    path = [f"random{i}" for i in range(6, 11)]
    
    for p in path:
        record_csv_path = f'logs/CartPole-v1/{p}/train_performance/training_record.csv'
        ds_record_result = pd.read_csv(record_csv_path, header=None)
        # seed = int(data_dir[-1])
        y_values = ds_record_result[1].values
        all_ys = []
        all_ys.append(y_values)
        all_ys = np.array(all_ys)

        xs = list(range(1, len(y_values)+1))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, y_values, label=f'seed {p}', alpha=0.7)
        ax.plot(xs, all_ys.mean(axis=0), label='mean', linestyle='--')
        ax.set_title(f'CartPole-v1 {p} ')
        ax.legend()
        ax.grid(axis='y')
        ax.set_facecolor('whitesmoke')
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Fitness', fontsize=16)
        plt.savefig(f'logs/CartPole-v1/{p}/train_performance/plot.png')
        plt.savefig(f'{p}.png')
        
        # # print the average learning performance
        # print(f' the average training performance of the last training record is {y_values[-1]} ')
        
    plt.show()


if __name__ == '__main__':
    main()
