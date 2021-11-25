import pandas as pd
import matplotlib.pyplot as plt
import os


# todo
# 1. training loss + test loss, training error + test error
# 2. accuracy ?
# 3. add legend

def generate_report(file_path):
    """ Creates training report based on saved logs """
    df = pd.read_csv(f'{file_path}.csv', sep=',')
    epochs = df['epoch']
    loss = df['loss']
    train_error = df['train_error']

    plt.plot(epochs, loss, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{file_path}_loss_fig.jpg')

    plt.plot(epochs, train_error, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.savefig(f'{file_path}_error_fig.jpg')


def log_results(file_path, df):
    if not os.path.isfile(f'{file_path}.csv'):
        df.to_csv(f'{file_path}.csv',
                  header=['epoch', 'loss', 'instance_error', 'bag_error'], index=False)
    else:
        df.to_csv(f'{file_path}.csv', mode='a', header=False)
