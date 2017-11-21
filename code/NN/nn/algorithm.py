class LearningRate(object):
    # Класс для вычисления шага обучения нейронной сети.

    # Параметры:
    #    const: float
    #        Постоянный шаг обучения.
    #    epoch: int
    #        Текущий номер эпохи, обновленный с помощью метода: `NN.train`.

    def __init__(self, const, epoch=0):
        self.const = const
        self.epoch = epoch

    def lr(self, t):
        return self.const / (t // 100 + 1.)

    def get(self):
        return self.lr(self.epoch)
