# 
# Provides a list of test cases to test each method 
# and the running of the neural network in general.
#
# Compiled against Python 2.7
# Author: Stephen Bahr (sbahr@bu.edu)

"""Unit tests for assignlayer[row] = p
          print layer[row]
          layer[row] = 1.0 / (1 + pow(2.7182818284, -1 * layer[row]))ment 4."""
import random
import unittest

from perceptron import Perceptron, NotConverged
from neural_network import NeuralNetwork

XOR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)]

OR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)]

def make_multiple_outputs(data):
  return [(features, [output]) for features, output in data]

def clip(v):
  """Convert a neural network output to a 0/1 prediction."""
  return 0 if v < 0.5 else 1

class Test(unittest.TestCase):
  longMessage=True

  def setUp(self):
    random.seed(1)

  def test_perceptron_predict(self):
    self.assertEqual(Perceptron([1, 2, 0]).predict([0, 0]), 0)
    self.assertEqual(Perceptron([1, 2, 0]).predict([1, 1]), 1)
    self.assertEqual(Perceptron([1, 2, 0]).predict([1, -1]), 0)
    self.assertEqual(Perceptron([1, 2, 0]).predict([-1, 1]), 1)
    self.assertEqual(Perceptron([1, 2, -4]).predict([1, 1]), 0)
    self.assertEqual(Perceptron([1, 2, -3]).predict([1, 1]), 0)
    self.assertEqual(Perceptron([1, 2, -2]).predict([1, 1]), 1)

  def test_perceptron_learn_xor(self):
    self.assertRaises(NotConverged, Perceptron().learn, XOR)

  def test_perceptron_learn_or(self):
    data = OR
    model = Perceptron().learn(data)
    for x, y in data:
      self.assertEqual(model.predict(x), y)

  _SIMPLE_MODEL = NeuralNetwork([
          [
          # weights into hidden unit 0
            [0.3, 0.4],
          # weights into hidden unit 1
            [-0.5, -0.6],
          ],
          # weights into output unit
          [[0.1, 0.2]]])
  _COMPLEX_MODEL = NeuralNetwork([
          [
            # weights into hidden unit 0
            [0.3, 0.4],
            # weights into hidden unit 1
            [-0.5, -0.6],
            # weights into hidden unit 2
            [0.9, -1.0],
          ],
          [
            # weights into output unit 0
            [0.1, 0.2, 1.1],
            # weights into output unit 1
            [0.7, -0.8, -1.2],
          ]])
  def test_get_unit_values_simple(self):
    inputs = [0, 0]
    unit_values = self._SIMPLE_MODEL.get_unit_values(inputs)
    self.assertEqual(len(unit_values), 3)  # three layers of units
    self.assertEqual(unit_values[0], inputs)  # layer 0 is inputs
    self.assertEqual(len(unit_values[1]), 2)  # layer 1 is hidden units
    self.assertAlmostEqual(unit_values[1][0], 0.5)
    self.assertAlmostEqual(unit_values[1][1], 0.5)
    self.assertEqual(len(unit_values[2]), 1)  # layer 2 is output units
    self.assertAlmostEqual(unit_values[2][0], 0.5374, places=4)

  def test_get_unit_values_complex(self):
    inputs = [-1.4, 1.3]
    unit_values = self._COMPLEX_MODEL.get_unit_values(inputs)
    self.assertEqual(len(unit_values), 3)  # three layers of units
    self.assertEqual(unit_values[0], inputs)  # layer 0 is inputs
    self.assertEqual(len(unit_values[1]), 3)  # layer 1 is hidden units
    self.assertAlmostEqual(unit_values[1][0], 0.5250, places=4)
    self.assertAlmostEqual(unit_values[1][1], 0.4800, places=4)
    self.assertAlmostEqual(unit_values[1][2], 0.0718, places=4)
    self.assertEqual(len(unit_values[2]), 2)  # layer 2 is output units
    self.assertAlmostEqual(unit_values[2][0], 0.5566, places=4)
    self.assertAlmostEqual(unit_values[2][1], 0.4744, places=4)

  def test_calculate_errors_simple(self):
    unit_values = self._SIMPLE_MODEL.get_unit_values([0, 0])
    errors = self._SIMPLE_MODEL.calculate_errors(unit_values, [0])
    self.assertEqual(len(errors), 2)  # Hidden errors, output errors
    self.assertEqual(len(errors[0]), 2)  # One error per hidden unit
    self.assertAlmostEqual(errors[0][0], -0.0033, places=4)
    self.assertAlmostEqual(errors[0][1], -0.0067, places=4)
    self.assertEqual(len(errors[1]), 1)  # One error per output unit
    self.assertAlmostEqual(errors[1][0], -0.1336, places=4)

  def test_learn_or(self):
    data = make_multiple_outputs(OR)
    model = NeuralNetwork().learn(data)
    for x, (y,) in data:
      self.assertEqual(clip(model.predict(x)[0]), y,
          msg='datum %s, %s' % (x, y))

  # def test_learn_xor(self):
  #   data = make_multiple_outputs(XOR)
  #   model = NeuralNetwork().learn(data)
  #   for x, (y,) in data:
  #     self.assertEqual(clip(model.predict(x)[0]), y,
  #         msg='datum %s, %s' % (x, y))


if __name__ == '__main__':
  unittest.main()
