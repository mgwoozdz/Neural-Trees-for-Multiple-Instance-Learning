def train(dataloader, model, optimizer, objective_fn, error_fn):
    model.train()

    train_loss = 0.
    train_error = 0.

    for bag, y_true, _ in dataloader:
        optimizer.zero_grad()                       # reset gradients
        y_prob, y_hat, _ = model.forward(bag)       # forward pass

        loss = objective_fn(y_prob, y_true)         # calculate loss
        train_loss += loss

        error = error_fn(y_hat, y_true)             # and error
        train_error += error

        loss.backward()                             # backward pass
        optimizer.step()                            # update params

    # mean for epoch
    train_loss /= len(dataloader)
    train_error /= len(dataloader)

    return float(train_loss), float(train_error)


def test(dataloader, model, objective_fn, error_fn):
    model.eval()

    test_loss = 0.
    test_error = 0.

    for bag, y_true, _ in dataloader:
        y_prob, y_hat, _ = model.forward(bag)       # inference

        loss = objective_fn(y_prob, y_true)         # calculate loss
        test_loss += loss

        error = error_fn(y_hat, y_true)             # and error
        test_error += error

    # mean for epoch
    test_loss /= len(dataloader)
    test_error /= len(dataloader)

    return float(test_loss), float(test_error)
