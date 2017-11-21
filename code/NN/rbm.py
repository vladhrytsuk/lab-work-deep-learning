# Ограниченная машина Больцмана.

from __future__ import division
import numpy as np
import _pickle as pkl

from scipy.special import expit

from nn.algorithm import LearningRate
from nn.utils import generate_batches


class RBM(object):
    # Класс ограниченной машины Больцмана (RBM).

    # Простая RBM с видимыми и скрытыми слоями.
    # Вывод выполняется с использованием контрастной дивергенции (CD-k).
    # Так можно использовать постоянную контрастную расходимость.

    # Параметры:
    #    n_visible: int
    #         Количество видимых (входных) слоев.
    #    n_hidden: int
    #         Количество скрытых слоев.
    #    k: int
    #         Количество шагов выборки Гиббса в алгоритме CD-k.
    #    persistent: bool
    #         Если "True" (по умолчанию), использует предыдущий отрицательный образец как
    #         начальные видимые слои для цепи Гиббса. В противном случае,
    #         новая цепочка Gibbs начинается с новой партии данных.
    #    learn_rate: float или .nn.algorithm.LearningRate
    #         Функция, которая принимает текущий номер эпохи и возвращает
    #         скорость обучения. Вход "float" становится "const".
    #         По умолчанию: lambda epoch: const / (epoch // 100 + 1.).
    #    early_stopping: boolean
    #         Если "True" (по умолчанию), попытка остановить обучение
    #         прежде чем ошибка проверки начнет увеличиваться.
    #    seed: float
    #         Случайный seed для инициализации и выборки.

    # Параметры без ввода:
    #    W: numpy.ndarray
    #         Матрица весов размером [n_hidden] x [n_visible].
    #    b: numpy.ndarray
    #         Вектор смещения для скрытых слоев размером [n_hidden] x 1.
    #    c: numpy.ndarray
    #         Вектор смещения для видимых слоев размером [n_visible] x 1.
    #    rng: numpy.random.RandomState
    #         Генератор случайных чисел NumPy с использованием "seed".

    # Методы:
    #    __init__, save, load, train, generate_negative_sample,
    #    sample_from_posterior, sample_from_likelihood, compute_posterior,
    #    compute_likelihood, compute_cross_entropy   

    def __init__(self, n_visible=784, n_hidden=100, k=1, persistent=True, learning_rate=0.1, early_stopping=True,
                 seed=99):
        # Инициализация модели RBM.

        # Параметры
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.persistent = persistent
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.seed = seed

        # Создание экземпляры классов "activation" и "learning_rate"
        if not isinstance(self.learning_rate, LearningRate):
            self.learning_rate = LearningRate(self.learning_rate)

        # Инициализация весов (аналогично как в модели "NN")
        self.rng = np.random.RandomState(seed)
        u = np.sqrt(6. / (n_hidden + n_visible))
        self.W = self.rng.uniform(-u, u, size=(n_hidden, n_visible))
        self.b = np.zeros((n_hidden, 1))
        self.c = np.zeros((n_visible, 1))

        # Постоянная инициализация цепочки
        if self.persistent:
            self.X_neg = None

        # Обновление обучения
        self.epoch = 0
        self.training_error = []
        self.validation_error = []

    def save(self, path):
        # Сохранение текущей модели в "path".
        with open(path, 'w') as f:
            pkl.dump(self, f)

    @staticmethod
    def load(path):
        # Загрузка модели, сохраненной функцией "save".
        with open(path) as f:
            rbm = pkl.load(f)
        if isinstance(rbm, RBM):
            return rbm
        else:
            raise Exception('Загруженный объект не является объектом "RBM".')

    def train(self, X, X_valid=None, batch_size=20, n_epoch=50, batch_seed=127, verbose=True):
        # Обучение RBM с помощью контрастной дивергенции (CD-k).

        # Параметры:
        #    X: numpy.ndarray (binary)
        #         Матрица входных данных размера [n] x [p], где "n" - размер выборки, а "p" - размер данных.
        #    X_valid: numpy.ndarray (binary)
        #         Дополнительная матрица данных проверки. Если это предусмотрено,
        #         текущая ошибка проверки сохраняется в модели.
        #    batch_size: int
        #         Размер случайных партий входных данных.
        #    n_epoch: int
        #         Количество эпох для обучения по входным данным.
        #    batch_seed: int
        #         Первый случайный seed для выбора партии.
        #    verbose: bool
        #         Если значение "True" (по умолчанию), обновите подготовку отчетов за эпоху до stdout.

        # Returns:
        #     rbm: RBM
        #         Обученная модель "RBM".

        assert self.W.shape[1] == X.shape[1]
        if X_valid is not None:
            assert X.shape[1] == X_valid.shape[1]
        elif self.early_stopping:
            raise Exception('RBM.train: нет данных проверки для ранней остановки.')
        n = X.shape[0]
        n_batches = int(np.ceil(n / batch_size))

        if self.persistent and self.X_neg is None:
            self.X_neg = self.rng.binomial(1, 0.5,
                                           size=(batch_size, self.n_visible))

        if verbose:
            print('|-------|---------------------------|---------------------------|')
            print('| Epoch |         Training          |         Validation        |')
            print('|-------|---------------------------|---------------------------|')
            print('|   #   |       Cross-Entropy       |       Cross-Entropy       |')
            print('|-------|---------------------------|---------------------------|')

        for t in range(n_epoch):

            for i, batch in enumerate(generate_batches(n, batch_size, batch_seed + t)):

                # Получение "X" в текущей партии и ее негативные примеры
                X_batch = X[batch, :]
                if self.persistent:
                    X_neg = self.generate_negative_sample(self.X_neg)
                else:
                    X_neg = self.generate_negative_sample(X_batch)
                n_batch = X_batch.shape[0]

                # Выполнение контрастных изменений градиента дивергенции
                lr = self.learning_rate.get()
                p_batch = self.compute_posterior(X_batch)
                p_neg = self.compute_posterior(X_neg)
                grad_W = (1. / n_batch) * (p_batch.T.dot(X_batch) - p_neg.T.dot(X_neg))
                grad_b = np.mean(p_batch - p_neg, axis=0, keepdims=True).T
                grad_c = np.mean(X_batch - X_neg, axis=0, keepdims=True).T
                self.W += lr * grad_W
                self.b += lr * grad_b
                self.c += lr * grad_c

                if self.persistent:
                    self.X_neg = X_neg

            self.epoch += 1
            self.learning_rate.epoch = self.epoch

            # Ошибка кросс-энтропии на основе стохастических реконструкций
            # с использованием обновленных параметров
            training_error = self.compute_cross_entropy(X)
            self.training_error.append((self.epoch, training_error))

            if X_valid is not None:
                validation_error = self.compute_cross_entropy(X_valid)
                self.validation_error.append((self.epoch, validation_error))
                if verbose:
                    print('|  {:3d}  |         {:9.5f}         |         {:9.5f}         |'. \
                          format(self.epoch, training_error, validation_error))
                if self.early_stopping:
                    if (self.epoch >= 100 and
                                    1.02 * min(self.validation_error[-2][1],
                                               self.validation_error[-3][1],
                                               self.validation_error[-4][1],
                                               self.validation_error[-5][1],
                                               self.validation_error[-6][1]) < validation_error):
                        print('======Early stopping: validation error increase at epoch {:3d}====='. \
                              format(self.epoch))
                        break
            else:
                if verbose:
                    print('|  {:3d}  |         {:9.5f}         |                           |'. \
                          format(self.epoch, training_error))

        if verbose:
            print('|-------|---------------------------|---------------------------|')

        return self

    def generate_negative_sample(self, X):
        # Создание негативных примеров (\tlide{x}), соответствующих
        # каждой точке данных с использованием шаг "k" выборки Гиббса.

        # Промежуточные образцы Гиббса не сохраняются, поскольку они
        # считаются обжигающими примерами.

        # Параметры:
        #    X: numpy.ndarray (binary)
        #          Матрица входных данных размера [n] x [n_visible],
        #          где "n" - размер выборки, а "n_visible" - размер входных данных.

        # Returns:
        #    X_neg: numpy.ndarray (binary)
        #          Матрица негативных примеров, соответствующих каждой точке данных в "X"
        #          размером [n] x [n_visible], где "n" - размер выборки, а "n_visible" - размер входных данных.

        assert self.n_visible == X.shape[1]

        X_new = X.copy()
        for _ in range(self.k):
            H_new = self.sample_from_posterior(X_new)
            X_new = self.sample_from_likelihood(H_new)

        return X_new

    def sample_from_posterior(self, X):
        # Пример из последующего распределения, заданного "X", т.е.
        # h_i ~ p(h_i | x_i) для каждого i = 1, ..., n.      

        # Дает один образец из каждой строки "X".
        # Используется предположение о независимости между скрытыми слоями.

        # Параметры:
        #     X: numpy.ndarray (binary)
        #           Матрица входных данных размера [n] x [n_visible],
        #           где "n" - размер выборки, а "n_visible" - размер входных данных.

        # Returns:
        #     H: numpy.ndarray (binary)
        #          Матрица скрытых переменных размера [n] x [n_hidden],
        #          где "n" - размер выборки, а "n_hidden" - количество скрытых единиц.

        P = self.compute_posterior(X)
        return self.rng.binomial(1, P)

    def sample_from_likelihood(self, H):
        # Пример из функции правдоподобия, заданной «H», т.е.
        # x_i ~ p(x_i | h_i) для каждого i = 1, ..., n.

        # Дает один пример из каждой строки "H".
        # Используется предположение о независимости между видимыми слоями.

        # Параметры:
        #     H: numpy.ndarray (binary)
        #          Матрица скрытых переменных размера [n] x [n_hidden],
        #          где "n" - размер выборки, а "n_hidden" - количество скрытых единиц.

        # Returns:
        #     X: numpy.ndarray (binary)
        #          Матрица новый примеров размера [n] x [n_visible],
        #          где "n" - размер выборки, а "n_visible" - размер входных данных.

        L = self.compute_likelihood(H)
        return self.rng.binomial(1, L)

    def compute_posterior(self, X):
        # Вычислить последущую вероятность
        # p (h_j = 1 | x) обусловлен данными "X".

        # Это вывод "исходящего потока" в RBM.

        # Параметры:
        #      X: numpy.ndarray (binary)
        #           Матрица входных данных размера [n] x [n_visible],
        #           где "n" - размер выборки, а "n_visible" - размер входных данных.

        # Returns:
        #      P: numpy.ndarray
        #           Матрица последующих вероятностей для каждой из точек данных n.
        #           размером [n] x [n_hidden], где "n" - размер выборки, а "n_hidden" - количество скрытых единиц.

        assert self.n_visible == X.shape[1]

        return expit(X.dot(self.W.T) + self.b.T)

    def compute_likelihood(self, H):
        # Вычисление функции правдоподобия
        # p(x_k = 1 | h) скрытых слоев "H" (каждая строка отличается от "h").

        # Это вывод "входящего потока" вывод в RBM.

        # Параметры:
        #     H: numpy.ndarray (binary)
        #          Матрица скрытых переменных размера [n] x [n_hidden],
        #          где "n" - размер выборки, а "n_hidden" - количество скрытых единиц.

        # Returns:
        #     L: numpy.ndarray
        #          Матрица вероятностей для каждого из скрытых единичных векторов "n"
        #          размера [n] x [n_visible], где "n" - размер выборки, а "n_visible" - размер входных данных.

        assert self.n_hidden == H.shape[1]

        return expit(H.dot(self.W) + self.c.T)

    def compute_cross_entropy(self, X):
        # Вычисление ошибки кросс-энтропии (отрицательный логарифмический правдоподобие)
        # между "X" и вероятностью текущей модели для этих данных.

        # Это дает основную оценку для RBM.
        # Ошибка суммируется по количеству видимых единиц и
        # масштабируется по размеру выборки (т. е. делится на "n"). Это позволяет
        # сравнить ошибки обучения и проверки в одном масштабе.

        # Параметры:
        #     X: numpy.ndarray
        #          Матрица входных данных для прогнозирования размера [n] x [n_visible],
        #          где "n" - размер выборки, а "n_visible" - размер данных.

        # Returns:
        #     error: float
        #           Средняя ошибка кросс-энтропии стохастической реконструкции.

        assert self.n_visible == X.shape[1]
        H = self.sample_from_posterior(X)
        L = self.compute_likelihood(H)  # нормализованный между [0, 1]
        return -(X * np.log(L + 1e-8) + (1 - X) * np.log((1 - L) + 1e-8)).sum(axis=1).mean()
