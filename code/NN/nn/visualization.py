# Инструменты визуализации (MNIST)

import numpy as np
import matplotlib.pyplot as plt


def print_image(X, output_shape=None, title=''):
    # Показывает набор 2D-изображений размером d * d.

    # Параметры:
    #    X: numpy.ndarray
    #         Матрица данных размера [n] x [d**2], где каждая строка соответствует
    #         черно-белому изображению размера [d] x [d], который сглаживается
    #         строгим упорядочиванием.
    #    output_shape: tuple of int (m = n_row, k = n_col)
    #         Необязательный аргумент, определяющий, в какой форме будут показаны
    #         изображения "n". Если "None", изображения будут отображены по горизонтали.
    #         Если "(m, k)", где "n == m * k", изображения "n" будут отображены в
    #         строгом порядке.

    # Returns:
    #     None

    n, dsq = X.shape
    if n > 400:
        raise Exception('print_image: слишком много входных изображений (более 400)')
    d = np.sqrt(dsq)
    if np.round(d) == d:
        d = int(d)
    else:
        raise Exception('print_image: входное изображение не квадратное')
    if output_shape is not None:
        m, k = output_shape
        assert n == m * k
    else:
        m, k = n, 1

    fig = plt.figure(figsize=(k, m)) 
    plt.gray()
    for i in range(m*k):
        ax = fig.add_subplot(m, k, i+1)
        # row-major
        ax.matshow(X[i, :].reshape(d, d))
        ax.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title)
    return fig
