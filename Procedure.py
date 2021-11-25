class Procedure(object):

    def __init__(self, model=None, train_loader=None, test_loader=None, optimizer=None, cuda=False):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.cuda = cuda

    def train(self, epoch):
        self.model.train()
        train_loss = 0.
        train_instance_error = 0.
        incorrect = 0
        for batch_idx, (data, label) in enumerate(self.train_loader):
            bag_label = label[0]
            if self.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = data, bag_label

            # reset gradients
            self.optimizer.zero_grad()
            # calculate loss and metrics
            loss, _ = self.model.calculate_objective(data, bag_label)
            train_loss += loss.data[0]
            error, predicted_label = self.model.calculate_classification_error(data, bag_label)
            train_instance_error += error
            incorrect += int(predicted_label[0][0]) != int(bag_label)
            # backward pass
            loss.backward()
            # step
            self.optimizer.step()
        # calculate loss and error for epoch
        train_loss /= len(self.train_loader)
        train_instance_error /= len(self.train_loader)
        train_bag_error = incorrect / len(self.train_loader)

        print('Epoch: {}, Loss: {:.4f}, Train instance error: {:.4f} Train bag error: {:.4f}'.format(
            epoch, train_loss.cpu().numpy()[0], train_instance_error, train_bag_error))

        return float(train_loss), float(train_instance_error), float(train_bag_error)

    def test(self):
        self.model.eval()
        test_loss = 0.
        test_instance_error = 0.
        incorrect = 0

        for batch_idx, (data, label, _) in enumerate(self.test_loader):
            bag_label = label[0]
            if self.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = data, bag_label
            loss, attention_weights = self.model.calculate_objective(data, bag_label)
            test_loss += loss.data[0]
            error, predicted_label = self.model.calculate_classification_error(data, bag_label)
            test_instance_error += error
            incorrect += int(predicted_label[0][0]) != int(bag_label)

        test_loss /= len(self.test_loader)
        test_instance_error /= len(self.test_loader)
        test_bag_error = incorrect / len(self.test_loader)

        print('\nTest Set, Loss: {:.4f}, Test instance error: {:.4f} Test bag error: {:.4f}'.format(
            test_loss.cpu().numpy()[0], test_instance_error, test_bag_error))

        return float(test_loss), float(test_instance_error), float(test_bag_error)
