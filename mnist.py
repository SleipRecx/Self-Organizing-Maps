from som import MSOM

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == "__main__":
    mnist = input_data.read_data_sets("data/mnist", one_hot=True)
    som = MSOM(cases=mnist.validation.images, labels=mnist.validation.labels)
    som.train()
    predictions = np.eye(10)[som.predict(mnist.validation.images)]
    print("train:", np.count_nonzero(np.equal(predictions, mnist.validation.labels)) / len(mnist.validation.labels))
    predictions = np.eye(10)[som.predict(mnist.test.images[0:1000])]
    print("test", np.count_nonzero(np.equal(predictions, mnist.test.labels[0:1000])) / len(mnist.test.labels[0:1000]))



