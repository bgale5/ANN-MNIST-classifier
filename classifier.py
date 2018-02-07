from network import *
from pprint import pprint


MINIBATCH_SIZE = 20
LEARNING_RATE = 3
EPOCHS = 30
NETWORK_INPUT_SIZE = 784
NETWORK_HIDDEN_SIZE = 30
NETWORK_OUTPUT_SIZE = 10
TRAINING_X = 'TrainDigitX.csv.gz'
TRAINING_Y = 'TrainDigitY.csv.gz'
TESTING_X = 'TestDigitX.csv.gz'
TESTING_Y = 'TestDigitY.csv.gz'


def load_csv(examples_file_name, labels_file_name):
    print("Loading training examples...")
    examples = np.genfromtxt(examples_file_name, delimiter=',', dtype=float)
    print("Loading labels...")
    labels = np.genfromtxt(labels_file_name, delimiter='\n', converters={0: vectorize})
    print("Done!")
    return [list(a) for a in zip(examples, labels)]


# Converts a integer to a vector representation
# eg. 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
def vectorize(y):
    y = int(y)
    if y not in range(0, 10):
        raise ValueError("Input must be an integer number between 0 and 9")
    label_vector = np.zeros((10, 1))
    label_vector[y] = 1
    return label_vector


# Partition the data set into n-sized minibatches
# Yields a generator to save memory and avoid copying all the data
def partition(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def learn(training_data):
    nn = Network(NETWORK_INPUT_SIZE, NETWORK_HIDDEN_SIZE, NETWORK_OUTPUT_SIZE, LEARNING_RATE)
    for epoch in range(EPOCHS):
        for minibatch in partition(training_data, MINIBATCH_SIZE):
            split = list(zip(*minibatch))  # zip splits data into separate examples and labels vectors
            x = np.column_stack(split[0])
            y = np.column_stack(split[1])
            nn.propagate_forward(x)
            print(nn.cost(y, nn.a3))
            nn.propagate_backward(x, y)
            nn.update_weights()
        np.random.shuffle(training_data)


td = load_csv(TRAINING_X, TRAINING_Y)
learn(td)
