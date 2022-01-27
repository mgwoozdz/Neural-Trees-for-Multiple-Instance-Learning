import torch as t
from torch import optim
from torch.autograd import Variable

from models.abmil import SAVE_PATH, AttentionBasedMIL
from utils.colon_cancer import kth_train_val_test_data_loaders
from utils.kfold_cross_val import train_val_test_split_warwick
from utils.helpers import get_device

device = get_device()

print(f"Device: {device}")

K_FOLDS = 10

train_folds, val_folds, test_folds = train_val_test_split_warwick(100, K_FOLDS, seed=2)

EPOCHS = 100

BATCH_SIZE = 1

model = AttentionBasedMIL()

data_loaders_0 = kth_train_val_test_data_loaders(
    train_folds, val_folds, test_folds, 0, batch_size=BATCH_SIZE
)


def train(model, optimizer, train_loader):
    train_loss = 0.0
    train_error = 0.0
    model.train(True)
    for batch_idx, (data, label) in enumerate(train_loader):
        label = label[0]
        data, label = Variable(data), Variable(label)
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        loss, _ = model.calculate_objective(data, label)
        train_loss += loss.item()
        error, _ = model.calculate_classification_error(data, label)
        train_error += error
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    return train_loss, train_error


def evaluate(model, data_loader):
    model.eval()
    evaluate_loss = 0.0
    evaluate_error = 0.0
    for batch_idx, (data, label) in enumerate(data_loader):
        label = label[0]
        data = data.to(device)
        label = label.to(device)
        data, label = Variable(data), Variable(label)
        loss, _ = model.calculate_objective(data, label)
        evaluate_loss += loss.item()
        error, _ = model.calculate_classification_error(data, label)
        evaluate_error += error

    evaluate_loss /= len(data_loader)
    evaluate_error /= len(data_loader)
    return evaluate_loss, evaluate_error


def experiment_cross_val():
    for k in range(K_FOLDS):
        model = AttentionBasedMIL()
        model = model.to(device)

        data_loaders = kth_train_val_test_data_loaders(
            train_folds, val_folds, test_folds, k, batch_size=BATCH_SIZE
        )

        optimizer = optim.Adam(
            model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
        print(f"Split: {k}")

        for epoch in range(EPOCHS):
            train_loss, train_error = train(model, optimizer, data_loaders["train"])
            eval_loss, eval_error = evaluate(model, data_loaders["val"])
            print(
                f"Epoch {epoch}; Train loss: {train_loss}; Train error: {train_error};"
                f"Val loss: {eval_loss}; Val error: {eval_error};"
            )
            scheduler.step()
        test_loss, test_error = evaluate(model, data_loaders["test"])
        print(f"Test loss: {test_loss}; Test error: {test_error}")


def experiment(model, save=False):
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=0.00001, betas=(0.9, 0.999), weight_decay=0.0005
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    for epoch in range(EPOCHS):
        train_loss, train_error = train(model, optimizer, data_loaders_0["train"])
        eval_loss, eval_error = evaluate(model, data_loaders_0["val"])
        print(
            f"Epoch {epoch}; Train loss: {train_loss}; Train error: {train_error};"
            f"Val loss: {eval_loss}; Val error: {eval_error};"
        )
        scheduler.step()
    test_loss, test_error = evaluate(model, data_loaders_0["test"])
    print(f"Test loss: {test_loss}; Test error: {test_error}")
    if save:
        t.save(model, SAVE_PATH)


if __name__ == "__main__":
    experiment(model, save=True)
