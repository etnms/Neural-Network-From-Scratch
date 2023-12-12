            
import matplotlib.pyplot as plt

def plot_data(data, label, x_label, y_label, title):
    plt.plot(data, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
