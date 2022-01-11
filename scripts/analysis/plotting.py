import matplotlib.pyplot as plt
import numpy as np


def plot(x_data, y_data, x_label, y_label, title, legend):
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(legend)
    plt.show()
