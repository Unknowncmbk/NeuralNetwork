# 
# Mimics an artificial perceptron, which can learn a from a 
# set of data.
#
# Compiled against Python 2.7
# Author: Stephen Bahr (sbahr@bu.edu)

import common


class NotConverged(Exception):
  """An exception raised when the perceptron training isn't converging."""


class Perceptron:
  def __init__(self, weights=None):
    self.weights = weights

  def learn(self, examples, max_iterations=100):
    """Learn a perceptron from [([feature], class)].

    Set the weights member variable to a list of numbers corresponding
    to the weights learned by the perceptron algorithm from the training
    examples.

    The number of weights should be one more than the number of features
    in each example.

    Args:
      examples: a list of pairs of a list of features and a class variable.
        Features should be numbers, class should be 0 or 1.
      max_iterations: number of iterations to train.  Gives up afterwards

    Raises:
      NotConverged, if training did not converge within the provided number
        of iterations.

    Returns:
      This object
    """

    #Initialize weights to 0
    self.weights = []
    length = 0;
    #For each tuple
    for element in examples:
      #input
      i = element[0]
      length = len(i)

    #n + 1 weights
    for x in range(0, length + 1):
      self.weights = self.weights + [0]

    #While not converged and max_iterations
    while not self.isConverged(examples) and max_iterations >= 0:
      #for each input
      for (x,y) in examples:
        #compute p
        p = self.predict(x)
        if not len(x) == len(self.weights):
          x = x + [1]
        if not p == y:
          if p < y:
            common.scale_and_add(self.weights, 1, x)
          else:
            common.scale_and_add(self.weights, -1, x)
          max_iterations = max_iterations - 1

    if not self.isConverged(examples):
      raise NotConverged()

    return self

  def predict(self, features):
    """Return the prediction given perceptron weights on an example.

    Args:
      features: A vector of features, [f1, f2, ... fn], all numbers

    Returns:
      1 if w1 * f1 + w2 * f2 + ... * wn * fn + t > 0
      0 otherwise
    """
    if not len(self.weights) == len(features):
      features = features + [1]

    if common.dot(self.weights, features) > 0:
      return 1
    else:
      return 0

  """ Returns true if the data works """
  def isConverged(self, examples):

    for element in examples:
      #input
      i = element[0]
      #output
      o = element[1]
      #if not enough features, add 1 to the features
      if not len(self.weights) == len(i):
        i = i + [1]

      if not self.predict(i) == o:
        return False

    return True
