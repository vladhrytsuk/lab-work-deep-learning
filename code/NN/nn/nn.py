# Модели нейронных сетей с обратной связью.

from __future__ import division
import numpy as np
import _pickle as pkl

from nn.layer import HiddenLayer, OutputLayer
from nn.activation import Activation
from nn.algorithm import LearningRate
from nn.utils import transform_y, generate_batches


class NN(object):
    # Класс нейронной сети.

    # Полностью связанные одномерные или многомерные нейронные сети
    # с нелинейными функциями активации, распада и исключений.
    # Предполагается, что выходной слой является функцией softmax.

    # Параметры:
    #     architecture: list of int
    #         Список целых чисел, которые содержат количество нейронов на слой.
    #         Например, "(784, 100, 100, 10)" указывает слой с двумя скрытыми слоями NN
    #         с 784-мерными входами, 100 скрытых нейронов в каждом из
    #         два скрытых слоя и 10-мерный выход.
    #     activation: string or Activation
    #         Выбор функции активации.
    #         Либо "sigmoid" (по умолчанию), "tanh", "relu", или "linear", или
    #         соответствующий экземпляр класса "Activation".
    #     learning_rate: function(epoch) or float
    #         Функция, которая принимает текущий номер эпохи и возвращает
    #         скорость обучения. Значение по умолчанию: lambda epoch: .1 / (epoch% 200 + 1.).
    #     momentum: float (от 0 до 1)
    #         Параметр импульса для экспоненциального усреднения предыдущего
    #         градиенты.
    #     weight_decay: float
    #         Параметр регуляции веса / L2. По умолчанию "1e-4".
    #     dropout: float (от 0 до 1)
    #         Вероятность выпадения единицы. По умолчанию "0.5".
    #     early_stopping: boolean
    #         Если "True" (по умолчанию), попытки остановить обучение
    #         прежде, чем ошибка проверки начнет увеличиваться.
    #     seed: float
    #         Случайное seed для инициализации и исключения.

    # Параметры без ввода:
    #     layers: list of "Layer"
    #         Список слоев, составляющих нейронную сеть. Слой включает
    #         входящую весовую матрицу и вектор смещения к ее единицам.
    #         Первый уровень соединяет вход с первым набором скрытых
    #         единиц, а последний слой соединяет последний набор скрытых единиц
    #         к выходным вероятностям (т.е. softmax).

    # Методы:
    #     __init__, save, load, train, predict, compute_error, compute_cross_entropy

    def __init__(self, architecture=[784, 100, 10], activation='sigmoid', learning_rate=0.1, momentum=0.5,
                 weight_decay=1e-4, dropout=0.5, early_stopping=True, seed=99):
        # Инициализация модели нейронной сети.

        # Параметры
        self.architecture = architecture
        self.activation = activation
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.seed = seed

        # Создание экземпляры классов "activation" и "learning_rate"
        if not isinstance(self.activation, Activation):
            self.activation = Activation(self.activation)
        if not isinstance(self.learning_rate, LearningRate):
            self.learning_rate = LearningRate(self.learning_rate)

        # Инициализация списка слоев нейронной сети
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(architecture[:-2],
                                              architecture[1:-1])):
            l = HiddenLayer('layer{}'.format(i), n_in, n_out, self.activation,
                            self.learning_rate, self.momentum,
                            self.weight_decay, self.dropout, self.seed + i)
            self.layers.append(l)

        # Выходной слой
        n_in, n_out = architecture[-2], architecture[-1]
        l = OutputLayer('output_layer', n_in, n_out,
                        self.learning_rate, self.momentum,
                        self.weight_decay, self.dropout, self.seed + i + 1)
        self.layers.append(l)

        # Обновление обучения
        self.epoch = 0
        self.training_error = []
        self.validation_error = []
        self.training_loss = []
        self.validation_loss = []

    def save(self, path):
        # Сохранение текущей модели в "path".
        with open(path, 'w') as f:
            pkl.dump(self, f)

    @staticmethod
    def load(path):
        # Загрузка модели, сохраненной функцией "save".
        with open(path) as f:
            rbm = pkl.load(f)
        if isinstance(nn, NN):
            return nn
        else:
            raise Exception('Загруженный объект не является объектом "NN".')

    def train(self, X, y, X_valid=None, y_valid=None, batch_size=200, n_epoch=40, batch_seed=0, verbose=True):
        # Обучение нейронной сети с данными.

        # Параметры:
        #     X: numpy.ndarray
        #         Матрица входных данных размера [n] x [p], где "n" - размер выборки, а "p" - размер данных.
        #     y: numpy.ndarray (binary)
        #         Матрица входных меток данных размера [n] x [k], где "n" - размер выборки, а "k" - количество классов.
        #     X_valid: numpy.ndarray
        #         Дополнительная матрица данных проверки. Если предоставляется "y_valid",
        #         текущая частота ошибок проверки сохраняется в модели.
        #     y_valid: numpy.ndarray
        #         Дополнительный вектор результатов проверки. Если предоставлено "X_valid",
        #         текущая частота ошибок проверки сохраняется в модели.
        #     batch_size: int
        #         Размер случайных партий входных данных.
        #     n_epoch: int
        #         Количество эпох для обучения по входным данным.
        #     batch_seed: int
        #         Первый случайный seed для выбора партии.
        #     verbose: bool
        #         Если значение "True" (по умолчанию), обновите подготовку отчетов за эпоху до stdout.

        # Returns:
        #     nn: NN
        #         Обученная модель "NN".

        assert self.layers[0].n_in == X.shape[1]
        assert X.shape[0] == y.shape[0]
        n = X.shape[0]
        n_batches = int(np.ceil(n / batch_size))

        if verbose:
            print('|-------|---------------------------|---------------------------|')
            print('| Epoch |         Training          |         Validation        |')
            print('|-------|---------------------------|---------------------------|')
            print('|   #   |    Error    |  Cross-Ent  |    Error    |  Cross-Ent  |')
            print('|-------|---------------------------|---------------------------|')

        for t in range(n_epoch):

            for i, batch in enumerate(generate_batches(n, batch_size, batch_seed + t)):

                # Прямое распространение (последний h - вероятность выхода)
                h = X[batch, :]
                for l in self.layers:
                    h = l.fprop(h, update_units=True)

                # Обратное распространение
                grad = -(y[batch, :] - h)
                for l in self.layers[::-1]:
                    grad = l.bprop(grad)

            self.epoch += 1
            for l in self.layers:
                l.learning_rate.epoch = self.epoch

            # Ошибки
            training_error = self.compute_error(X, y)
            training_loss = self.compute_cross_entropy(X, y)
            self.training_error.append((self.epoch, training_error))
            self.training_loss.append((self.epoch, training_loss))

            if X_valid is not None and y_valid is not None:
                validation_error = self.compute_error(X_valid, y_valid)
                validation_loss = self.compute_cross_entropy(X_valid, y_valid)
                self.validation_error.append((self.epoch, validation_error))
                self.validation_loss.append((self.epoch, validation_loss))
                if verbose:
                    print('|  {:3d}  |   {:.5f}   |   {:.5f}   |   {:.5f}   |   {:.5f}   |'. \
                          format(self.epoch, training_error, training_loss, validation_error, validation_loss))
                if self.early_stopping:
                    if (self.epoch >= 40 and
                                self.validation_loss[-2][1] < validation_loss and
                                self.validation_loss[-3][1] < validation_loss and
                                self.validation_loss[-4][1] < validation_loss):
                        print('======Early stopping: validation loss increase at epoch {:3d}======'. \
                              format(self.epoch))
                        break
            else:
                if verbose:
                    print('|  {:3d}  |     {:.5f}    |      {:.5f}       |               |                   |'. \
                          format(self.epoch, training_error, training_loss))

        if verbose:
            print('|-------|---------------------------|---------------------------|')

        return self

    def predict(self, X, output_type='response'):
        # Предсказание метки, используя текущие параметры модели.

        # Параметры:
        #     X: numpy.ndarray
        #        Матрица входных данных размером [n] x [p] для прогнозирования,
        #        где "n" - размер выборки, а "p" - размер данных.
        #     output_type: string
        #        Тип выхода.
        #        "response" (по умолчанию) возвращает предсказанный "y" как вектор one-hot,
        #        "prob" или "probability" возвращает вероятность softmax.

        #  Returns:
        #     Одно из следующего:
        #       y: numpy.ndarray (binary)
        #            Матрица прогнозируемых меток размером [n] x [c]
        #            в one-hot формате, если "output_type", является "response" (по умолчанию),
        #            где "n" - размер выборки, а "c" - количество классов.
        #       p: numpy.ndarray
        #            Матрица прогнозируемых softmax вероятностей размером [n] x [c],
        #            если "output_type", является "prob" или "probability",
        #            где "n" - размер выборки, а "c" - количество классов.

        assert self.layers[0].n_in == X.shape[1]

        h = X
        for l in self.layers:
            h = l.fprop(h)

        if output_type == 'response':
            return transform_y(np.argmax(h, axis=1), h.shape[1])
        elif output_type == 'prob' or output_type == 'probability':
            return h
        else:
            raise Exception('NN.predict: нераспознанный "output_type"')

    def compute_error(self, X, y):
        # Вычисление коэффициента ошибок для "X" и "y".

        # Параметры:
        #    X: numpy.ndarray
        #         Матрица входных данных размером [n] x [p] для прогнозирования,
        #         где "n" - размер выборки, а "p" - размер данных.
        #    y: numpy.ndarray (binary)
        #         Матрица меток данных размером [n] x [с] в one-hot формате,
        #         где "n" - размер выборки, а "c" - количество классов.

        # Returns:
        #    err: float
        #         Коэффициент ошибок прогноза.

        assert X.shape[0] == y.shape[0]
        return 1. - np.all(self.predict(X) == y, axis=1).mean()

    def compute_cross_entropy(self, X, y):
        # Вычисляет потерю кросс-энтропии (отрицательное логарифмическое правдоподобие)
        # между "y" и "self.predict(X)".

        # Параметры:
        #    X: numpy.ndarray
        #         Матрица входных данных размером [n] x [p] для прогнозирования,
        #         где "n" - размер выборки, а "p" - размер данных.
        #    y: numpy.ndarray (binary)
        #         Матрица меток данных размером [n] x [с] в one-hot формате,
        #         где "n" - размер выборки, а "c" - количество классов.

        # Returns:
        #    loss: float
        #         Средняя потеря поперечной энтропии по данным.

        assert X.shape[0] == y.shape[0]
        p = self.predict(X, 'prob')

        return -(y * np.log(p + 1e-8)).mean()
