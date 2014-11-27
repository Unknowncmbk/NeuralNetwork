# 
# Mimics an artificial neural network, which is a set of perceptrons. 
# We can make this neural network learn from a set of data, and 
# then predict on a set of examples. 
#
# Compiled against Python 2.7
# Author: Stephen Bahr (sbahr@bu.edu)

import collections
import common
import math
import random
import sys

# Throughout this file, layer 0 of a neural network is the inputs, layer 1
# is the first hidden layer, etc.; the last layer is the outputs.

class NeuralNetwork:
  """An artificial neural network.

  Fields:
    weights: a list of lists of lists of numbers, where
       weights[a][b][c] is the weight into unit b of layer a+1 from unit c in
         layer a
    num_hidden_layers: an integer, the number of hidden layers in the network
  """

  def __init__(self, weights=None):
    self.weights = weights
    if weights:
      self.num_hidden_layers = len(weights) - 1

  def get_unit_values(self, features):
    """Calculate the activation of each unit in a neural network.

    Args:
      features: a vector of feature values

    Returns:
      units, a list of lists of numbers, where
        units[a][b] is the activation of unit b in layer a
    """
    #Initialize the list of lists
    lst = []

    # iterate over the entire network
    for x in range(0, self.num_hidden_layers + 2):
      # if first layer
      if x == 0:
        # add the inputs
        lst.append(features)
      # else if end layer
      elif x == self.num_hidden_layers + 1:
        # add empty base, number is based off last arg of weights
        end = []
        for y in range(0, len(self.weights[1])):
          end.append(0)
        lst.append(end)
      # middle layer
      else:
        # Number of nodes in this layer x is the length
        # weights[0]
        mid = []
        for y in range(0, len(self.weights[0])):
          mid.append(0)
        lst.append(mid)

    # For each layer
    for column in range(0, len(lst)):
      # if the last layer
      if column == len(lst) - 1:
        # for each row in that layer
        for row in range(0, len(lst[column])):
          layer = lst[column]
          weights = self.weights[1][row]
          p = 0
          # for each value i in the hidden layer to the left
          for i in range(0, len(lst[column - 1])):
            w = weights[i]
            p = p + (w * lst[column-1][i])
          layer[row] = self.activation(p)
      elif column > 0:
        # for each row in that layer
        for row in range(0, len(lst[column])):
          layer = lst[column]
          weights = self.weights[0][row]
          p = 0
          # for each value i in the hidden layer to the left
          for i in range(0, len(lst[column - 1])):
            w = weights[i]
            p = p + (w * lst[column-1][i])
          layer[row] = self.activation(p)

    return lst


  def predict(self, features):
    """Calculate the prediction of a neural network on one example

    Args:
      features: a vector of feature values

    Returns:
      A list of numbers, the predictions for each output of the network
          for the given example.
    """
    return self.get_unit_values(features)[1]


  def calculate_errors(self, unit_values, outputs):
    """Calculate the backpropagated errors for an input to a neural network.

    Args:
      unit_values: unit activations, a list of lists of numbers, where
        unit_values[a][b] is the activation of unit b in layer a
      outputs: a list of correct output values (numbers)

    Returns:
      A list of lists of numbers, the errors for each hidden or output unit.
          errors[a][b] is the error for unit b in layer a+1.
    """

    errors = self.constructUnits(unit_values[0])
    for layer in range(len(errors)-1, 0, -1):
      # if the last layer
      if layer == len(errors) - 1:
        # for each parent in that layer
        for parent in range(0, len(errors[layer])):
          p_out = unit_values[layer][parent]
          e_out = p_out * (1 - p_out) * (outputs[parent] - p_out)
          errors[layer][parent] = e_out
      elif layer > 0:
        # for each parent in that layer
        for parent in range(0, len(errors[layer])):
          err = 0
          for child in range(0, len(self.weights[layer])):
            w_hidden_to_out = self.weights[layer][child][parent]
            p_hidden = unit_values[layer][parent]
            err_out = errors[layer+1][child]
            err = err + p_hidden * (1-p_hidden) * (w_hidden_to_out * err_out)

          errors[layer][parent] = err

    return errors[1:]

  def activation(self, v):
    return 1 / (1 + math.exp(-v))

  def learn(self,
      data,
      num_hidden=16,
      max_iterations=2000,
      learning_rate=1,
      num_hidden_layers=1):
    """Learn a neural network from data.

    Sets the weights for a neural network based on training data.

    Args:
      data: a list of pairs of input and output vectors, both lists of numbers.
      num_hidden: the number of hidden units to use.
      max_iterations: the max number of iterations to train before stopping.
      learning_rate: a scaling factor to apply to each weight update.
      num_hidden_layers: the number of hidden layers to use.
        Unless you are doing the extra credit, you can ignore this parameter.

    Returns:
      This object, once learned.
    """

    self.initializeWeights(data, num_hidden, num_hidden_layers)
    while max_iterations > 0:
      max_iterations = max_iterations - 1

      for d in data:
        i = d[0]
        o = d[1]
        units = self.get_unit_values(i)
        errors = self.calculate_errors(units, o)
        print "Errors: " + str(errors)

        # For every layer of weights
        for layer in range(1, len(self.weights) + 1):
          # For each parent in the layer to the left (aka input)
          for parent in range(0, len(self.weights[layer - 1])):
            #print "Node: " + str(parent)
            for child in range(0, len(self.weights[layer - 1][parent])):
              self.weights[layer - 1][parent][child] += learning_rate * units[layer-1][child] * errors[layer - 1][parent]

    return self

  def constructUnits(self, features):
    """Construct the neural net units from the number of hidden layers

    Returns:
      A list of lists
    """
    lst = []
    for x in range(0, self.num_hidden_layers + 2):
      # if first layer
      if x == 0:
        # add the inputs
        lst.append(features)
      # else if end layer
      elif x == self.num_hidden_layers + 1:
        # add empty base, number is based off last arg of weights
        end = []
        for y in range(0, len(self.weights[1])):
          end.append(0)
        lst.append(end)
      # middle layer
      else:
        # Number of nodes in this layer x is the length
        # weights[0]
        mid = []
        for y in range(0, len(self.weights[0])):
          mid.append(0)
        lst.append(mid)
    return lst

  def initializeWeights(self, data, num_hidden, num_hidden_layers):
    weights = []
    hidden = []

    for column in range(0, num_hidden_layers):
      w = []
      if column == 0:
        for row in range(0, num_hidden):
          wei = []
          # For each input set in the first layer
          for input in range(0, len(data[0][0])):
            # For each element in that tuple
            wei.append(random.random())
            # for item in range(0, len(data[input])):
            #   wei.append(random.random())
          w.append(wei)
      else:
        for input in range(0, num_hidden):
          w.append(0)
      hidden = w

    output = []
    for column in range(0, len(data[0][1])):
      o = []
      for i in range(0, num_hidden):
        o.append(random.random())
      output.append(o)

    weights.append(hidden)
    weights.append(output)

    self.weights = weights
    self.num_hidden_layers = num_hidden_layers

    return
