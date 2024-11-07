import numpy as np
from torchvision.datasets import MNIST
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


def relu(z):
    return np.maximum(0, z)


def he_init(size1, size2):
    w = np.random.randn(size1, size2) * np.sqrt(2 / size1)
    b = np.zeros((1, size2))
    return w, b


def categorical_cross_entropy(y, A3):
    m = y.shape[0]
    log_likelihood = -np.log(A3[range(m), np.argmax(y, axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss


def softmax_derivative(A):
    return A * (1 - A)


def categorical_cross_entropy_derivative(A, y):
    m = A.shape[0]
    eroare = A - y
    return eroare / m


def relu_derivative(Z):
    return Z > 0


# Forward propagation
def forward_propagation(X, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, w3) + b3
    a3 = softmax(z3)
    return a1, a2, a3


def update_weights_and_bias(w1, b1, w2, b2, w3, b3, learning_rate, dw1, dw2, dw3, db1, db2, db3):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    return w1, b1, w2, b2, w3, b3


def test_model(test_X, test_Y, w1, b1, w2, b2, w3, b3):
    a1, a2, a3 = forward_propagation(test_X, w1, b1, w2, b2, w3, b3)

    predictions = np.argmax(a3, axis=1)
    true_labels = np.argmax(test_Y, axis=1)

    accuracy = np.mean(predictions == true_labels)
    print(f"Acuratețea pe setul de testare: {accuracy * 100:.2f}%")


def train_model(batch_X, batch_Y, w1, b1, w2, b2, w3, b3, learning_rate):
    a1, a2, a3 = forward_propagation(batch_X, w1, b1, w2, b2, w3, b3)
    eroare = categorical_cross_entropy(batch_Y, a3)

    dz3 = categorical_cross_entropy_derivative(a3, batch_Y)
    dw3 = np.dot(a2.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)

    dz2 = np.dot(dz3, w3.T) * relu_derivative(a2)
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, w2.T) * relu_derivative(a1)
    dw1 = np.dot(batch_X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    w1, b1, w2, b2, w3, b3 = update_weights_and_bias(
        w1, b1, w2, b2, w3, b3, learning_rate, dw1, dw2, dw3, db1, db2, db3
    )
    return eroare


input_size = 784
hidden_layer1_size = 64
hidden_layer2_size = 36
output_size = 10
learning_rate = 0.01
nr_epochs = 200
batch_size = 100

w1, b1 = he_init(input_size, hidden_layer1_size)
w2, b2 = he_init(hidden_layer1_size, hidden_layer2_size)
w3, b3 = he_init(hidden_layer2_size, output_size)


def scheduler_learning_rate(medie_eroare_old, medie_eroare_new):
    global learning_rate
    if medie_eroare_new >= medie_eroare_old:
        print("Changed learning rate")
        learning_rate = learning_rate * 0.1

def init_model():
    global learning_rate
    start_time = time.time()
    medie_eroare_old = 1
    medie_eroare_epoci_adunate = 0
    for epoch in range(nr_epochs):
        print(f"Epoch: {epoch + 1}/{nr_epochs}")
        batches_X, batches_Y = split_in_batches(train_X, train_Y_encoded, batch_size)
        eroare_totala = 0
        for batch_X, batch_Y in zip(batches_X, batches_Y):
            eroare_batch = train_model(batch_X, batch_Y, w1, b1, w2, b2, w3, b3, learning_rate)
            eroare_totala += eroare_batch
        medie_eroare_epoca = eroare_totala / len(batches_X)
        medie_eroare_epoci_adunate += medie_eroare_epoca
        if (epoch+1) % 5 == 0:
            medie_eroare_new = medie_eroare_epoci_adunate/5
            scheduler_learning_rate(medie_eroare_old, medie_eroare_new)
            medie_eroare_old = medie_eroare_new
            medie_eroare_epoci_adunate = 0

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Timpul total de antrenare: {training_time:.2f} secunde")

    print("Testare")
    test_model(test_X, test_Y_encoded, w1, b1, w2, b2, w3, b3)


init_model()
