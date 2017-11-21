import numpy as np


class Layer(object):
    # Базовый класс для слоев нейронной сети.

    # Включает весовую матрицу и вектор смещения, а также
    # гиперпараметры для обучения.

    # Параметры (поступает из `NN.__init__`):
    #    __name__: string
    #           Название слоя, например `layer3`.
    #    n_in: int
    #           Количество входных единиц слоя.
    #    n_out: int
    #           Количество выходных единиц слоя.
    #    learning_rate: function(epoch) or float
    #           Функция, которая принимает текущий номер эпохи и возвращает
    #           скорость обучения.
    #    momentum: float (от 0 до 1)
    #           Параметр импульса для экспоненциального усреднения предыдущего
    #           градиенты.
    #    weight_decay: float
    #           Вес распада / L2, параметр регуляризации.
    #    dropout: float (от 0 до 1)
    #           Вероятность выпадения единицы.
    #    seed: float
    #           Случайное величина для инициализации и исключения.

    # Параметры без ввода:
    #    W: numpy.ndarray
    #           Весовая матрица размером [n_out] x [n_in].
    #    b: numpy.ndarray
    #           Вектор смещения размером [n_out] x 1.
    #    rng: numpy.random.RandomState
    #           Генератор случайных чисел с использованием `seed`.

    # Методы:
    #    __init__, fprop, bprop

    def __init__(self, name, n_in, n_out, learning_rate, momentum, weight_decay, dropout, seed):
        # Инициализация слоя нейронной сети.

        # Параметры
        self.__name__ = name
        self.n_in = n_in
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed

        # Инициализация весов и смещения
        self.rng = np.random.RandomState(seed)
        c = np.sqrt(6. / (n_in + n_out))
        self.W = self.rng.uniform(-c, c, size=(n_out, n_in))
        self.b = np.zeros((n_out, 1))

        # Сохраненные значения для обратного распространения
        self.h_in = None
        self.a_out = None

    def fprop(self, h, update_units=False):
        # Метод прямого распространения. Реализовано в подклассе.
        raise NotImplementedError()

    def bprop(self, grad):
        # Метод обратного распространения. Реализовано в подклассе.
        raise NotImplementedError()


class HiddenLayer(Layer):
    # Подкласс "Layer" для скрытых слоев.
    # cм. подкласс "OutputLayer" для построения выходного слоя.

    # Дополнительные параметры:
    #    activation: Activation
    #           Экземпляр класса Activation.

    # Дополнительные параметры без ввода:
    #    h_in, a_out: numpy.ndarray
    #           Значения промежуточной единицы во время обратного распространения.

    # Методы:
    #    __init__, fprop, bprop

    def __init__(self, name, n_in, n_out, activation, learning_rate, momentum, weight_decay, dropout, seed):
        # Инициализация скрытого слоя нейронной сети.

        # Инициализация суперкласса
        super(HiddenLayer, self).__init__(name, n_in, n_out, learning_rate, momentum, weight_decay, dropout, seed)

        # Градиенты (изначально 0, заданные матрицами для расчета momentum (импульса))
        self.grad_a_out = np.zeros((2, self.n_out))  # подразумевает среднеквадратичное значение
        self.grad_W = 0.0
        self.grad_decay = 0.0
        self.grad_b = 0.0
        self.grad_h_in = np.zeros((2, self.n_in))  # подразумевает среднеквадратичное значение

        # Дополнительные параметры
        self.activation = activation
        self.h_in = None
        self.a_out = None

    def fprop(self, h_in, update_units=False):
        # Прямое распространение входящих единиц через текущий слой.
        # Включает линейное преобразование и активацию.
        # Может принимать пакетные входы размера [n] x 1.

        # Параметры:
        #     h_in: numpy.ndarray
        #         Матрица размером [n] x [n_in], которая соответствует
        #         скрытым единицам из входящего потока.
        #         Каждой строке соответствует скрытое значение единицы из
        #         каждой точки данных в текущей партии.
        #     update_units: boolean
        #         Если "True", сохраняет "h_in" и "a_in", которые позже используются для
        #         обратное распространение. Параметр "False" по умолчанию используется для
        #         прогнозирование.

        # Returns:
        #     h_out: numpy.ndarray
        #         Матрица размером [n] x [n_out], которая соответствует
        #         активированным скрытым единицам в исходящем потоке.
        #         Каждой строке соответствует скрытое значение единицы из
        #         каждой точки данных в текущей партии.

        assert isinstance(h_in, np.ndarray)
        assert h_in.shape[1] == self.n_in

        # Для каждой точки данных это "a_out = W.dot(h_in) + b"
        a_out = h_in.dot(self.W.T) + self.b.T

        if update_units:
            self.h_in = h_in
            self.a_out = a_out

        h_out = self.activation.eval(a_out)

        # Исключение
        if update_units:
            mask = self.rng.binomial(1, 1. - self.dropout, size=h_out.shape)
        else:
            # Ожидание во время тестирования
            mask = (1. - self.dropout) * np.ones(h_out.shape)
        return h_out * mask

    def bprop(self, grad_h_out):
        # Обратное распространение градиента w.r.t. после-активационных единиц
        # в исходящем потоке.
        # Обновляет параметры модели ("W" и "b") и возвращает градиент w.r.t.
        # после-активационные единицы во входящем потоке.

        # Для аргумента и возвращаемого значения каждая строка соответствует
        # градиенту от каждой точки данных в текущей партии.

        # Параметры:
        #    grad_h_out: numpy.ndarray
        #       Матрица градиентов размером [n] x [n_out] по отношению к
        #       после-активационным единицам в исходящем потоке.

        # Возвращает:
        #    grad_h_in: numpy.ndarray
        #        Матрица градиентов размером [n] x [n_in] по отношению к
        #        после-активационным единицам во входящем потоке.

        # Утвердить, что "fprop" уже выполнено.
        assert self.h_in is not None and self.a_out is not None
        assert self.h_in.shape[0] == grad_h_out.shape[0]
        assert self.a_out.shape[0] == grad_h_out.shape[0]
        n = self.h_in.shape[0]

        # Вычислить градиенты
        self.grad_a_out = grad_h_out * self.activation.grad(self.a_out) + self.momentum * self.grad_a_out.mean(axis=0)
        self.grad_W = (1. / n) * self.grad_a_out.T.dot(self.h_in) + self.momentum * self.grad_W
        self.grad_decay = 2. * self.weight_decay * self.W + self.momentum * self.grad_decay
        self.grad_b = self.grad_a_out.mean(axis=0, keepdims=True).T + self.momentum * self.grad_b
        self.grad_h_in = self.grad_a_out.dot(self.W) + self.momentum * self.grad_h_in.mean(axis=0)

        # Обновление стохастического градиентного спуска
        lr = self.learning_rate.get()
        self.W = self.W - lr * (self.grad_W + self.grad_decay)
        self.b = self.b - lr * self.grad_b

        return self.grad_h_in


class OutputLayer(Layer):
    # Подкласс "Layer" для выходного слоя (softmax).
    # cм. подкласс "HiddenLayer" для построения скрытого слоя.

    # Дополнительные параметры без ввода:
    #    h_in: numpy.ndarray
    #          Последние после-активационные единицы до softmax. Используется в backprop.

    # Методы:
    #    __init__, fprop, bprop

    def __init__(self, name, n_in, n_out, learning_rate, momentum, weight_decay, dropout, seed):
        # Инициализация выходного слоя нейронной сети.

        # Инициализация суперкласса
        super(OutputLayer, self).__init__(name, n_in, n_out, learning_rate, momentum, weight_decay, dropout, seed)

        # Градиенты (изначально 0, заданные матрицами для расчета momentum (импульса))
        self.grad_W = 0.0
        self.grad_decay = 0.0
        self.grad_b = 0.0
        self.grad_h_in = np.zeros((2, self.n_in))  # подразумевает среднеквадратичное значение

        # Дополнительные параметры
        self.h_in = None

    def fprop(self, h_in, update_units=False):
        # Прямое распространение последних скрытых единиц на слой softmax.
        # Включает линейную трансформацию и преобразование softmax.
        # Может принимать пакетные входы размера [n] x 1.

        # Параметры:
        #     h_in: numpy.ndarray
        #         Матрица размером [n] x [n_in], которая соответствует
        #         скрытым единицам из входящего потока.
        #         Каждой строке соответствует скрытое значение единицы из
        #         каждой точки данных в текущей партии.
        #     update_units: boolean
        #         Если "True", сохраняет "h_in" и "a_in", которые позже используются для
        #         обратное распространение. Параметр "False" по умолчанию используется для
        #         прогнозирование.

        # Returns:
        #     h_out: numpy.ndarray
        #         Матрица размером [n] x [n_out], которая соответствует
        #         оценочным вероятностям в выходном слое.
        #         Каждая строка соответствует оценочной вероятности для
        #         каждой точки данных в текущей партии.

        assert isinstance(h_in, np.ndarray)
        assert h_in.shape[1] == self.n_in

        # Для каждой точки данных это "a_out = W.dot(h_in) + b"
        a_out = h_in.dot(self.W.T) + self.b.T

        if update_units:
            self.h_in = h_in

        # Softmax
        ex = np.exp(a_out)
        h_out = ex / (ex.sum(axis=1, keepdims=True) + 1e-8)

        return h_out

    def bprop(self, grad_a_out):
        # Обратное распространение градиента w.r.t. пред-активационных (softmax)
        # скрытых единиц.
        # Обновляет параметры модели ("W" и "b") и возвращает градиент w.r.t.
        # после-активационные единицы во входящем потоке.

        # Для аргумента и возвращаемого значения каждая строка соответствует
        # градиенту от каждой точки данных в текущей партии.

        # Параметры:
        #    grad_a_out: numpy.ndarray
        #       Матрица градиентов размером [n] x [n_out] по отношению к
        #       пред-активационным (softmax) скрытым единицам в исходящем потоке.

        # Возвращает:
        #    grad_h_in: numpy.ndarray
        #        Матрица градиентов размером [n] x [n_in] по отношению к
        #        после-активационным единицам во входящем потоке.

        # Утвердить, что "fprop" уже выполнено.
        assert self.h_in is not None
        assert self.h_in.shape[0] == grad_a_out.shape[0]
        n = self.h_in.shape[0]

        # Вычислить градиенты и сохранить их для следующей итерации (momentum)
        self.grad_W = (1. / n) * grad_a_out.T.dot(self.h_in) + self.momentum * self.grad_W
        self.grad_decay = 2. * self.weight_decay * self.W + self.momentum * self.grad_decay
        self.grad_b = grad_a_out.mean(axis=0, keepdims=True).T + self.momentum * self.grad_b
        self.grad_h_in = grad_a_out.dot(self.W) + self.momentum * self.grad_h_in.mean(axis=0)

        # Обновление стохастического градиентного спуска
        lr = self.learning_rate.get()
        self.W = self.W - lr * (self.grad_W + self.grad_decay)
        self.b = self.b - lr * self.grad_b

        return self.grad_h_in


class AEInputLayer(HiddenLayer):
    # Подкласс "HiddenLayer" для входного слоя автоэнкодера.

    # Точно такие же функции, как "HiddenLayer", за исключением того, что градиенты
    # связанные весы выводятся на "bprop".

    def __init__(self, name, n_in, n_out, activation, learning_rate, momentum, weight_decay, dropout, seed):
        # Инициализая входного слоя автоэнкодера.

        # Инициализая суперкласса
        super(AEInputLayer, self).__init__(name, n_in, n_out, activation, learning_rate, momentum, weight_decay,
                                           dropout, seed)

    def bprop(self, grad_h_out):
        # Единственное отличие от "HiddenLayer" заключается в том, что обновления градиента
        # для привязанной весовой матрицы откладывается до основного цикла тренировки.

        # Параметры:
        #    grad_h_out: numpy.ndarray
        #       Матрица градиентов размером [n] x [n_hidden] по отношению к
        #       после-активационным скрытым единицам в исходящем потоке.

        # Returns:
        #    grad_h_in: numpy.ndarray
        #        Матрица градиентов размером [n] x [n_visible] по отношению к
        #        после-активационным единицам во входящем потоке.

        # Утвердить, что "fprop" уже выполнено.
        assert self.h_in is not None and self.a_out is not None
        assert self.h_in.shape[0] == grad_h_out.shape[0]
        assert self.a_out.shape[0] == grad_h_out.shape[0]
        n = self.h_in.shape[0]

        # Вычисление градиентов
        self.grad_a_out = grad_h_out * self.activation.grad(self.a_out) + self.momentum * self.grad_a_out.mean(axis=0)
        self.grad_W = (1. / n) * self.grad_a_out.T.dot(self.h_in) + self.momentum * self.grad_W
        self.grad_decay = 2. * self.weight_decay * self.W + self.momentum * self.grad_decay
        self.grad_b = self.grad_a_out.mean(axis=0, keepdims=True).T + self.momentum * self.grad_b
        self.grad_h_in = self.grad_a_out.dot(self.W) + self.momentum * self.grad_h_in.mean(axis=0)

        # Обновление стохастического градиентного спуска
        lr = self.learning_rate.get()
        self.b = self.b - lr * self.grad_b

        return self.grad_h_in


class AEOutputLayer(Layer):
    # Подкласс "Layer" для выходгого слоя автоэнкодера.
    # Аналогично "HiddenLayer", но принимает градиенты предварительной активации во время bprop.
    # Подробнее см. в классе ".autoencoder.Autoencoder".

    # Дополнительные входы:
    #    W: numpy.ndarray
    #          Связанная матрица весов размером [n_hidden] x [n_visible]
    #          от прямого "HiddenLayer" автоэнкодера.

    # Дополнительные атрибуты:
    #    activation: Activation
    #          Экземпляр класса Activation.

    # Дополнительные атрибуты без ввода:
    #    h_in, a_out: numpy.ndarray
    #          Значения промежуточной единицы во время обратного распространения.

    # Методы:
    #    __init__, fprop, bprop

    def __init__(self, name, n_in, n_out, W, activation, learning_rate, momentum, weight_decay, dropout, seed):
        # Инициализация выходного слоя автоэнкодера.

        # Инициализация суперкласса
        # n_in (n_hidden) и n_out (n_visible) являются избыточными
        super(AEOutputLayer, self).__init__(name, n_in, n_out, learning_rate, momentum, weight_decay, dropout, seed)

        # Связанные веса: "self.W.T" - это веса fprop.
        # Стоит обратить внимание, что смещение не разделяется и инициализируется случайным образом выше.
        self.W = W

        # Градиенты (изначально 0, заданные матрицами для расчета momentum (импульса))
        self.grad_a_out = np.zeros((2, self.n_out))  # подразумевает среднеквадратичное значение
        self.grad_W = 0.0
        self.grad_decay = 0.0
        self.grad_b = 0.0
        self.grad_h_in = np.zeros((2, self.n_in))  # подразумевает среднеквадратичное значение

        # Дополнительные параметры
        self.activation = activation
        self.h_in = None
        self.a_out = None

    def fprop(self, h_in, update_units=False):
        # Прямое распространение входящих единиц через текущий слой.
        # Включает линейное преобразование и активацию.
        # Может принимать пакетные входы размера [n] x 1.

        # Параметры:
        #     h_in: numpy.ndarray
        #         Матрица размером [n] x [n_hidden], которая соответствует
        #         скрытым единицам из входящего потока.
        #         Каждой строке соответствует скрытое значение единицы из
        #         каждой точки данных в текущей партии.
        #     update_units: boolean
        #         Если "True", сохраняет "h_in" и "a_in", которые позже используются для
        #         обратное распространение. Параметр "False" по умолчанию используется для
        #         прогнозирование.

        # Returns:
        #     h_out: numpy.ndarray
        #         Матрица размером [n] x [n_visible], которая соответствует
        #         активированным скрытым единицам в выходном слое.
        #         Каждой строке соответствует скрытое значение единицы из
        #         каждой точки данных в текущей партии.

        assert isinstance(h_in, np.ndarray)
        assert h_in.shape[1] == self.n_in

        # Для каждой точки данных это "a_out = W.T.dot(h_in) + b"
        a_out = h_in.dot(self.W) + self.b.T

        if update_units:
            self.h_in = h_in
            self.a_out = a_out

        h_out = self.activation.eval(a_out)

        # Исключение
        if update_units:
            mask = self.rng.binomial(1, 1. - self.dropout, size=h_out.shape)
        else:
            # Ожидание во время тестирования
            mask = (1. - self.dropout) * np.ones(h_out.shape)
        return h_out * mask

    def bprop(self, grad_a_out):
        # Обратное распространение градиента w.r.t. пред-активационных единиц
        # в исходящем потоке.
        # Обновляет параметры модели ("W" и "b") и возвращает градиент w.r.t.
        # после-активационные единицы во входящем потоке.

        # Для аргумента и возвращаемого значения каждая строка соответствует
        # градиенту от каждой точки данных в текущей партии.

        # Параметры:
        #    grad_a_out: numpy.ndarray
        #       Матрица градиентов размером [n] x [n_visible] по отношению к
        #       пред-активационным единицам в исходящем потоке.

        # Возвращает:
        #    grad_h_in: numpy.ndarray
        #        Матрица градиентов размером [n] x [n_hidden] по отношению к
        #        после-активационным единицам во входящем потоке.

        # Утвердить, что "fprop" уже выполнено.
        assert self.h_in is not None and self.a_out is not None
        assert self.h_in.shape[0] == grad_a_out.shape[0]
        n = self.h_in.shape[0]

        # Вычисление градиентов
        # Связанные градиенты веса используют транспонирование.
        self.grad_W = (1. / n) * self.h_in.T.dot(grad_a_out) + self.momentum * self.grad_W
        self.grad_decay = 2. * self.weight_decay * self.W + self.momentum * self.grad_decay
        self.grad_b = grad_a_out.mean(axis=0, keepdims=True).T + self.momentum * self.grad_b
        self.grad_h_in = grad_a_out.dot(self.W.T) + self.momentum * self.grad_h_in.mean(axis=0)

        # Обновление стохастического градиентного спуска
        # Фактическое обновление происходит в цикле "Autoencoder.train".
        lr = self.learning_rate.get()
        self.b = self.b - lr * self.grad_b

        return self.grad_h_in
