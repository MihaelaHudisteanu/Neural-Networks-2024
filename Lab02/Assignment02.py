import numpy as np
from torchvision.datasets import MNIST


# 1. Load the MNIST dataset
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
    return mnist_data, mnist_labels


# 2. Normalize the data and convert the labels to one-hot-encoding
# normalize from [0,255] pixel value to [0,1]
def normalize(x, max_value):
    return np.array(x).astype(np.float32) / max_value


# one-hot-encoding -> from value to vector
def one_hot_encode(y, num_classes):
    one_hot_y = np.zeros((len(y), num_classes))

    for i in range(len(y)):
        one_hot_y[i, y[i]] = 1

    return one_hot_y


def initialized_params(input_size, output_size):
    W = np.random.randn(input_size, output_size)
    b = np.zeros((output_size,))
    return W, b


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def forward(x, W, b):
    z = np.dot(x, W) + b
    return softmax(z)


def cross_entropy_loss(predictions, labels):
    return -np.sum(labels * np.log(predictions))


def backward(x, y_true, y_pred, W, b, learning_rate):
    W = W + learning_rate * np.dot(x.T, y_true - y_pred)
    # b = b + learning_rate * (y_true - y_pred)
    b = b + learning_rate * np.sum(y_true - y_pred, axis=0)

    return W, b


def train(x_train, y_train, W, b, epochs, batch_size=100, learning_rate=0.01):
    for epoch in range(epochs):
        permutation = np.random.permutation(x_train.shape[0])
        x_train = x_train[permutation]
        y_train = y_train[permutation]

        # split
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            y_pred = forward(x_batch, W, b)

            loss = cross_entropy_loss(y_pred, y_batch)

            W, b = backward(x_batch, y_batch, y_pred, W, b, learning_rate)

        if epoch % 10 == 0:
            print("Epoch " + str(epoch+10) + " has error: " + str(loss))

    return W, b


def accuracy(x, y, W, b):
    y_pred = forward(x, W, b)
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y, axis=1)
    return np.mean(predictions == labels)


if __name__ == "__main__":
    x_train, y_train = download_mnist(True)
    x_test, y_test = download_mnist(False)

    x_train = normalize(x_train, 255.0)
    x_test = normalize(x_test, 255.0)

    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)

    W, b = initialized_params(784, 10)

    test_acc = accuracy(x_test, y_test, W, b) * 100
    print('Test accuracy before training: ' + str(test_acc) + '%')

    W, b = train(x_train, y_train, W, b, epochs=100, learning_rate=0.01)

    test_acc = accuracy(x_test, y_test, W, b) * 100
    print('Test accuracy after training: ' + str(test_acc) + '%')