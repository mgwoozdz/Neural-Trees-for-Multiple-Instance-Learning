import pandas as pd
import matplotlib.pyplot as plt
import os


# todo
# 1. training loss + test loss, training error + test error
# 2. accuracy ?
# 3. add legend


def generate_report(file_path):
    """Creates training report based on saved logs"""
    df = pd.read_csv(f"{file_path}", sep=",")
    epochs = df["epoch"]
    loss = df["loss"]
    instance_error = df["instance_error"]
    bag_error = df["bag_error"]

    plt.plot(epochs, loss, "r--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{file_path}_loss_fig.jpg")

    plt.plot(epochs, instance_error, "b-")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.savefig(f"{file_path}_error_fig.jpg")


def log_results(file_path, df):
    print(f"(log_results) logging {df}")
    if not os.path.isfile(f"{file_path}.csv"):
        df.to_csv(
            f"{file_path}.csv",
            header=["epoch", "loss", "instance_error", "bag_error"],
            index=False,
        )
    else:
        df.to_csv(f"{file_path}.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    generate_report("TrainingLogs/test_logs_1_2021-11-25_19:52:34.002999.csv")
