# Модель автоэнкодера

from __future__ import division
import numpy as np
import _pickle as pkl

from nn.layer import AEInputLayer, AEOutputLayer
from nn.activation import Activation
from nn.algorithm import LearningRate
from nn.utils import generate_batches


class Autoencoder(object):
    # Класс автоэнкодера.

    # Простой автоэнкодер с входным слоем, скрытыми слоями и связанными весами.
    # Обучение проводится с использованием градиентного спуска.
    #
    # Параметры:
    #    n_visible: int
    #         Количество видимых (входных) слоев.
    #    n_hidden: int
    #         Количество скрытых слоев.
    #    binary: bool
    #         Независимо от того, являются ли входные слои двоичными.
    #    denaising: float (от 0 до 1)
    #         Коэффициент отсева для шумоподавления автоэнкодеров.
    #         Использование 0.0 (по умолчанию) дает простой автоэнкодер без шумоподавления.
    #    learn_rate: float или .nn.algorithm.LearningRate
    #         Функция, которая принимает текущий номер эпохи и возвращает
    #         скорость обучения. Вход "float" становится "const".
    #         По умолчанию: lambda epoch: const / (epoch // 100 + 1.).
    #    momentum: float (от 0 до 1)
    #         Параметр импульса для экспоненциального усреднения предыдущего градиенты.
    #    weight_decay: float
    #         Параметр регуляции веса / L2. По умолчанию "1e-4".
    #    early_stopping: boolean
    #         Если "True" (по умолчанию), попытка остановить обучение
    #         прежде чем ошибка проверки начнет увеличиваться.
    #    seed: float
    #         Случайный seed для инициализации и выборки.

    # Параметры без ввода:
    #    W: numpy.ndarray
    #         Матрица весов размером [n_hidden] x [n_visible].
    #    b: numpy.ndarray
    #         Вектор смещения для скрытых слоев размером [n_hidden] x 1.
    #    c: numpy.ndarray
    #         Вектор смещения для видимых слоев размером [n_visible] x 1.
    #    rng: numpy.random.RandomState
    #         Генератор случайных чисел NumPy с использованием "seed".
    #
    # Методы:
    #    __init__, save, load, train, reconstruct, compute_error

    def __init__(self, n_visible=784, n_hidden=100, binary=True, activation='sigmoid', denoising=0.0, learning_rate=0.1,
                 momentum=0.5, weight_decay=1e-4, early_stopping=True, seed=99):
        # Инициализация модели автоэнкодера.

        # Параметры
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.binary = binary
        self.activation = activation
        self.denoising = denoising
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.seed = seed

        # Создание экземпляры классов "activation" и "learning_rate"
        if not isinstance(self.activation, Activation):
            self.activation = Activation(self.activation)
        if not isinstance(self.learning_rate, LearningRate):
            self.learning_rate = LearningRate(self.learning_rate)

        # Скорость подавления шума
        assert 0.0 <= self.denoising <= 1.0

        # Входной слой
        encoder = AEInputLayer('AE_input_layer', n_visible, n_hidden, self.activation, self.learning_rate,
                               self.momentum, self.weight_decay, 0.0, self.seed + 1)

        # Указатели на случайные инициализированные веса и смещения
        self.W = encoder.W
        self.b = encoder.b

        # Выходной слой с привязанными весами (смещение различается: "self.c")
        decoder = AEOutputLayer('AE_output_layer', n_hidden, n_visible, self.W, self.activation, self.learning_rate,
                                self.momentum, self.weight_decay, 0.0, self.seed + 2)
        self.c = decoder.b
        self.layers = [encoder, decoder]

        # RNG для шумоподавления
        self.rng = np.random.RandomState(self.seed)

        # Обновление обучения
        self.epoch = 0
        self.training_error = []
        self.validation_error = []

    def save(self, path):
        # Сохранение текущей модели в "path".
        pkl.dump(self)

    @staticmethod
    def load(path_dir):
        # Загрузка модели, сохраненной функцией "save".
        rbm = pkl.load(fname)
        if isinstance(rbm, RBM):
            return rbm
        else:
            raise Exception('Загруженный объект не является объектом "RBM".')

    def train(self, X, X_valid=None, batch_size=200, n_epoch=40, batch_seed=127, verbose=True):
        # Обучение автоэнкодера с использованием обратного распространения.

        # Параметры:
        #    X: numpy.ndarray (binary)
        #          Матрица входных данных размера [n] x [p], где "n" - размер выборки, а "p" - размер данных.
        #    X_valid: numpy.ndarray (binary)
        #          Дополнительная матрица данных проверки. Если это предусмотрено,
        #          текущая ошибка проверки сохраняется в модели.
        #    batch_size: int
        #          Размер случайных партий входных данных.
        #    n_epoch: int
        #          Количество эпох для обучения по входным данным.
        #    batch_seed: int
        #          Первый случайный seed для выбора партии.
        #    verbose: bool
        #          Если значение "True" (по умолчанию), обновите подготовку отчетов за эпоху до stdout.

        # Returns:
        #    rbm: RBM
        #        Обученная модель "RBM".

        assert self.W.shape[1] == X.shape[1]
        if X_valid is not None:
            assert X.shape[1] == X_valid.shape[1]
        elif self.early_stopping:
            raise Exception('RBM.train: нет данных проверки для ранней остановки.')
        n = X.shape[0]
        n_batches = int(np.ceil(n / batch_size))

        if verbose:
            print('|-------|---------------------------|---------------------------|')
            print('| Epoch |         Training          |         Validation        |')
            print('|-------|---------------------------|---------------------------|')
            print('|   #   |       Cross-Entropy       |       Cross-Entropy       |')
            print('|-------|---------------------------|---------------------------|')

        for t in range(n_epoch):

            for i, batch in enumerate(generate_batches(n, batch_size, batch_seed + t)):

                # Прямое распространение (последний h - вероятность выхода)
                if self.denoising > 0.0:
                    x = X[batch, :]
                    mask = self.rng.binomial(1, 1. - self.denoising, size=x.shape)
                    h = x * mask
                else:
                    h = X[batch, :]
                for l in self.layers:
                    h = l.fprop(h, update_units=True)

                # Обратное распространение
                grad = -(X[batch, :] - h)  # кросс-энтропия или потеря квадрата
                for l in self.layers[::-1]:
                    grad = l.bprop(grad)

                # Обновление привязанных градиентов веса
                lr = self.learning_rate.get()
                for l in self.layers[::-1]:
                    self.W = self.W - lr * (l.grad_W + l.grad_decay)
                # Переназначить обновленные привязанные веса
                for l in self.layers:
                    l.W = self.W

                # Обновление смещения (SGD уже выполняется на каждом уровне)
                self.b = self.layers[0].b
                self.c = self.layers[1].b

            self.epoch += 1
            self.learning_rate.epoch = self.epoch
            for l in self.layers:
                l.learning_rate.epoch = self.epoch

            # Ошибка кросс-энтропии на основе стохастических реконструкций
            # с использованием обновленных параметров
            training_error = self.compute_error(X)
            self.training_error.append((self.epoch, training_error))

            if X_valid is not None:
                validation_error = self.compute_error(X_valid)
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

    def reconstruct(self, X, prob=False):
        # Восстановление "X" с помощью автоэнкодера (прямое распространение).

        # Параметры:
        #    X: numpy.ndarray
        #         Матрица входных данных размером [n] x [n_visible], где "n" - размер выборки,
        #         а "n_visible" - размер входных данных.
        #    prob: bool
        #         Это относится только к "self.binary == True".
        #         Если "True", возвращает вероятности сигмоида для каждого
        #         измерение. Если "False" (по умолчанию), возвращает пороговое значение
        #         двоичная реконструкция.

        # Returns:
        #    X_hat: numpy.ndarray
        #         Матрица реконструированных образцов, соответствующие каждой точке данных в "X"
        #         размером [n] x [n_visible], где "n" - размер выборки, а "n_visible" - размер входных данных.

        assert self.n_visible == X.shape[1]

        h = X
        for l in self.layers:
            h = l.fprop(h, update_units=False)

        if self.binary and not prob:
            return (h >= 0.5).astype(np.int8)
        else:
            return h

    def compute_error(self, X):
        # Вычисление ошибки (кросс-энтропия, если двоичная, или квадратичная потеря, если она действительна)
        # между "Х" и его реконструкцией по текущей модели.

        # Это дает основной показатель оценки для автоэнкодеров.
        # Стоит обратите внимание, что ошибка суммируется по количеству видимых слоев и
        # масштабируется по размеру выборки (т. е. делится на "n"). Это позволяет
        # сравнить ошибки обучения и проверки в одном масштабе.

        # Параметры:
        #    X: numpy.ndarray
        #         Матрица входных данных для прогнорирования размером [n] x [n_visible],
        #         где "n" - размер выборки, а "n_visible" - размер входных данных.

        # Returns:
        #    error: float
        #         Средняя ошибка кросс-энтропии стохастической реконструкции.

        X_hat = self.reconstruct(X, prob=True)
        if self.binary:
            return -(X * np.log(X_hat + 1e-8) + (1 - X) * np.log((1 - X_hat) + 1e-8)).sum(axis=1).mean()
        else:
            return ((X - X_hat) ** 2).sum(axis=1).mean()
