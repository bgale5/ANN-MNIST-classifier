import numpy as np


class Network:

    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        # Parameters
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b1 = np.random.randn(self.hidden_size, 1)
        self.b2 = np.random.randn(self.output_size, 1)
        # Propagation Variables
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None
        self.w1_gradient = None
        self.w2_gradient = None
        self.b1_gradient = None
        self.b2_gradient = None

    # The sigmoid activation function for hidden layer nodes
    @staticmethod
    def _logistic(z):
        return 1 / (1 + np.exp(-z))

    # Derivative of the sigmoid activation function, enables back propagation
    @staticmethod
    def _logistic_prime(z):
        return np.multiply(Network._logistic(z), 1 - Network._logistic(z))

    # Activation function for the outer layer
    # Normalises the neural network's output matrix so that the value of each class is its likelyhood percentage
    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # Derivative of the softmax function, enables backpropagation
    @staticmethod
    def _softmax_prime(z):
        f = lambda x: Network._softmax(x) * (1 - Network._softmax(x))
        return np.apply_along_axis(f, 0, z)

    # The cost (Mean Squared Error) function to be minimised through training iterations
    @staticmethod
    def cost(label, output, axis=None):
        return ((output - label) ** 2).mean(axis=axis)

    # Derivative of MSE function for backward pass
    @staticmethod
    def _cost_prime(y, x):
        return -1 * np.subtract(y, x)

    @staticmethod
    def _gradient(delta, a):
        return np.dot(delta, np.transpose(a))

    # Sum all the weighted inputs and add bias
    @staticmethod
    def _net_input(x, w, b):
        w_augmented = np.c_[np.transpose(w), b]
        x_augmented = np.r_[x, np.ones((1, x.shape[1]))]
        return np.dot(w_augmented, x_augmented)

    # Pass training data forward through each layer of the network
    def propagate_forward(self, x):
        self.z2 = self._net_input(x, self.w1, self.b1)
        self.a2 = Network._logistic(self.z2)
        self.z3 = self._net_input(self.a2, self.w2, self.b2)
        self.a3 = Network._softmax(self.z3)

    # Perform backward pass, calculating error gradient for each weight
    def propagate_backward(self, x, y):
        # Output layer
        delta3 = np.multiply(Network._cost_prime(y, self.a3), Network._softmax_prime(self.z3))
        self.w2_gradient = Network._gradient(delta3, self.a2)
        self.b2_gradient = np.mean(delta3, axis=1)
        # Hidden layer
        delta2 = np.multiply(np.dot(self.w2, delta3), Network._logistic_prime(self.z2))
        self.w1_gradient = Network._gradient(delta2, x)
        self.b1_gradient = np.mean(delta2, axis=1)

    # Tune each weight proportionally to its contribution to the total cost
    def update_weights(self):
        self.w2 = np.subtract(self.w2, self.learning_rate * np.transpose(self.w2_gradient))
        self.w1 = np.subtract(self.w1, self.learning_rate * np.transpose(self.w1_gradient))
        self.b2 = np.subtract(self.b2, np.row_stack(self.learning_rate * self.b2_gradient))
        self.b1 = np.subtract(self.b1, np.row_stack(self.learning_rate * self.b1_gradient))

