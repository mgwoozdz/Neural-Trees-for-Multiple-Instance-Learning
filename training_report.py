import pandas as pd
import matplotlib.pyplot as plt
import os


# todo
# 1. training loss + test loss, training error + test error
# 2. accuracy ?
# 3. add legend


def generate_report(train_file_path, test_file_path, out_file_path_prefix):
    """Creates training report based on saved logs"""
    train_df = pd.read_csv(f"{train_file_path}", sep=",")
    epochs = train_df["epoch"]
    train_loss = train_df["loss"]
    train_error = train_df["error"]

    val_df = pd.read_csv(f"{test_file_path}", sep=",")
    # val_epochs = val_df["epoch"]
    val_loss = val_df["loss"]
    val_error = val_df["error"]

    # plot loss
    plt.plot(epochs, train_loss, "r-", epochs, val_loss, "r--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{out_file_path_prefix}_loss_fig.jpg")

    plt.clf()

    # plot error
    plt.plot(epochs, train_error, "b-", epochs, val_error, "b--")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.savefig(f"{out_file_path_prefix}_error_fig.jpg")


def log_results(file_path, df):
    # print(f"(log_results) logging {df}")
    if not os.path.isfile(f"{file_path}.csv"):
        df.to_csv(
            f"{file_path}.csv",
            header=["epoch", "loss", "error"],
            index=False,
        )
    else:
        df.to_csv(f"{file_path}.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    generate_report('TrainingLogs/train_logs_1_2021-11-25_23:03:57.614965.csv', 'TrainingLogs/val_logs_1_2021-11-25_23:03:57.614965.csv', "Figures/test")
