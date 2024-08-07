import numpy as np


def activation(x):
    return 0 if x < 0.5 else 1


def go(cond1, cond2, cond3):
    x = np.array([cond1, cond2, cond3])
    first_hidden_neuron = [0.3, 0.3, 0]  # Определяем вес связи по отношении ко входным данным
    second_hidden_neuron = [0.4, 0.5, 1]
    weight1 = np.array([first_hidden_neuron, second_hidden_neuron])  # матрица 2х3, 2 - кол-во нейронов, 3 - кол-во связей
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, x)
    out_hidden = np.array([activation(value) for value in sum_hidden])

    sum_end = np.dot(weight2, out_hidden)
    y = activation(sum_end)
    return y


data = go(0, 0, 1)
print(data)
