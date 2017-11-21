import numpy as np
from scipy.special import expit


class Activation(object):
    # Класс функций активации для нейронных сетей.

    # Параметры:
    #    type: string
    #        Либо "sigmoid" (по умолчанию), "tanh", "relu", или "linear".
    # Методы:
    #    eval, grad

    def __init__(self, type_of_activation_function='sigmoid'):
        self.type = type_of_activation_function
        if self.type not in ['sigmoid', 'tanh', 'relu', 'linear']:
            raise Exception('Activation.__init__: ' + 'Activation type not recognized')

    def eval(self, a):
        # Оценка активации при значении "a" (векторизованное).

        if self.type == 'sigmoid':
            return expit(a)
        elif self.type == 'tanh':
            return np.tanh(a)
        elif self.type == 'relu':
            return a * (a > 0)
        else:  # linear
            return a

    def grad(self, a):
        # Вычисление градиента активации при значении "a" (векторизованное).

        if self.type == 'sigmoid':
            s = self.eval(a)
            return s * (1. - s)
        elif self.type == 'tanh':
            return 1. - self.eval(a) ** 2
        elif self.type == 'relu':
            return a > 0
        else:  # linear
            return 1.
