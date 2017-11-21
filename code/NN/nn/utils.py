from __future__ import division
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path_train, path_valid, path_test):
    # Загружает данные из трех файлов CSV.

    # Параметры:
    #    path_train, path_valid, path_test: string
    #          Относительные пути к наборам данных для обучения, проверки и тестирования.

    # Returns:
    #    X_train, X_valid, X_test, y_train, y_valid, y_test: numpy.ndarray
    #          Матрицы данных и исходные векторы для каждого из наборов данных.

    # Raises:
    #    IOError: ошибка при доступе к одному из указанных файлов.

    data_train = np.genfromtxt(path_train, delimiter=',')
    data_valid = np.genfromtxt(path_valid, delimiter=',')
    data_test = np.genfromtxt(path_test, delimiter=',')

    X_train, y_train = data_train[:, :-1], transform_y(data_train[:, -1].astype(int))
    X_valid, y_valid = data_valid[:, :-1], transform_y(data_valid[:, -1].astype(int))
    X_test, y_test = data_test[:, :-1], transform_y(data_test[:, -1].astype(int))

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def transform_y(y, n_classes=None):
    # Преобразует категориальный вектор результата в / из одного горячего вектора.

    # Параметры:
    #    y: numpy.ndarray
    #           "n" - вектор категориальных исходов.
    #           Записи являются одним из "0", "1", ..., "c-1".
    #         или
    #           Бинарная матрица размером [n] x [c].
    #           "y [i, k] == 1", если "i" точка данных принадлежит классу "k".
    #    n_classes: int
    #         Количество категориальных результатов. По умолчанию используется значение "max(y) + 1".

    # Returns:
    #    y: numpy.ndarray
    #         Другая форма ввода "y".

    if len(y.shape) == 1:
        if n_classes is None:
            n_classes = max(y) + 1
        return np.eye(len(y), n_classes, dtype=bool)[y, :]
    elif len(y.shape) == 2:
        return np.argmax(y, axis=1)
    else:
        raise Exception('utils.transform_y: ' + 'аргумент не является правильным вектором результата')


def binarize_data(X, threshold=0.5):
    # Бинаризация обучающих входов до 0 или 1.

    # Это используется для входов для ограниченных машин Больцмана.

    # Параметры:
    #    X: numpy.ndarray
    #        Матрица данных обучения масштабируется в пределах [0, 1].

    # Returns:
    #    X: numpy.ndarray (dtype: np.int8)
    #        Матрица данных обучения порождена на пороге ввода.

    return (X >= threshold).astype(np.int8)


def standardize_data(X):
    # Стандартизация входов обучения к нулевому среднему и единичному вариантам.
    #     
    # Обертка вокруг "sklearn.preprocessing.StandardScaler()".

    # Параметры:
    #    X: numpy.ndarray
    #         Матрица данных обучения

    # Returns:
    #    scaler: sklearn.preprocessing.StandardScaler
    #          Scaler установлен на входные данные обучения "X".
    #          Используется "scaler.fit(X_new)", чтобы преобразовать данные проверки или тестирования.
    #          Используется "scaler.mean_" и "scaler.scale_" для оригинального
    #          среднего и стандартного отклонения.

    return StandardScaler().fit(X)


def generate_batches(n, batch_size, batch_seed=None):
    # Генератор для партий.

    # Параметры:
    #    n: int
    #        Общее количество точек данных на выбор.
    #    batch_size: int
    #        Количество точек данных за партию.
    #    batch_seed: int
    #        Случайный seed для упорядочивания данных.

    # Returns:
    #    Генератор, который каждый раз выводит список индексов для партии.

    batch_size = min(batch_size, n)
    rng = np.random.RandomState(batch_seed)
    perm = rng.choice(np.arange(n), n, replace=False)
    for i in range(int(np.ceil(n / batch_size))):
        yield perm[i * batch_size:(i + 1) * batch_size]
