import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import time


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = train_X / 255.0
test_X = test_X / 255.0


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


train_Y_encoded = one_hot_encode(train_Y, 10)
test_Y_encoded = one_hot_encode(test_Y, 10)


def split_in_batches(train_X, train_Y, batch_size):
    indices = np.random.permutation(len(train_X))
    train_X_shuffled = train_X[indices]
    train_Y_shuffled = train_Y[indices]

    num_batches = int(np.ceil(len(train_X) / batch_size))
    batches_X = np.array_split(train_X_shuffled, num_batches)
    batches_Y = np.array_split(train_Y_shuffled, num_batches)
    return batches_X, batches_Y


def softmax(z):
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward_propagation(X, W, b):
    z = np.dot(X, W) + b
    y_hat = softmax(z)
    return y_hat


def predict(y_hat):
    return np.argmax(y_hat, axis=1)


def train_model(batch_X, batch_Y, W, b, learning_rate):
    for i in range(batch_X.shape[0]):
        y_hat = forward_propagation(batch_X[i:i + 1], W, b)

        target = batch_Y[i]
        error = (target - y_hat)

        W += learning_rate * np.dot(batch_X[i:i + 1].T, error)
        b += learning_rate * error.flatten()


weights = np.zeros((784, 10))
learning_rate = 0.0001
bias = np.zeros((10,))
nr_epochs = 100
batch_size = 100
start_time = time.time()
for epoch in range(nr_epochs):
    print("Epoch: {}/{}".format(epoch + 1, nr_epochs))
    batches_X, batches_Y = split_in_batches(train_X, train_Y_encoded, batch_size)
    for batch_X, batch_Y in zip(batches_X, batches_Y):
        train_model(batch_X, batch_Y, weights, bias, learning_rate)

end_time = time.time()
training_time = end_time - start_time
print(f"Timpul total de antrenare: {training_time:.2f} secunde")
def calculate_accuracy(predictions, true_labels):
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    accuracy = np.mean(predictions == true_labels)
    return accuracy


def test_model(test_X, test_Y, weights, bias):
    y_hat = forward_propagation(test_X, weights, bias)
    predictions = predict(y_hat)
    accuracy = calculate_accuracy(predictions, test_Y)
    print(f"Acuratețea pe setul de testare: {accuracy * 100:.2f}%")


test_model(test_X, test_Y, weights, bias)


def display_perceptrons(weights):
    num_classes = weights.shape[1]

    plt.figure(figsize=(10, 5))

    for i in range(num_classes):
        plt.subplot(2, 5, i + 1)
        perceptron_image = weights[:, i].reshape(28, 28)
        plt.imshow(perceptron_image, cmap='gray')
        plt.title(f'Perceptron {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


display_perceptrons(weights)
