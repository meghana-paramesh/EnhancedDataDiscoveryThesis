import matplotlib.pyplot as plt
import numpy as np


def plot_date_diff(diff_d, diff_f, diff_s):
    
    print("len_f: ", len(diff_f), " len_d", len(diff_d), "len_s: ", len(diff_s))

    i = len(diff_f)

    # while i<len(diff_s):
    #     diff_f.append(0)
    #     i=i+1

    # i=len(diff_d)

    # while i<len(diff_s):
    #     diff_d.append(0)
    #     i=i+1

    # abc = [diff_d, diff_f, diff_s]
    # colours = ['blue', 'orange', 'green']
    # for aa,c in zip(abc, colours):
    #     aaa = np.array(aa)
    #     scat = plt.scatter(aaa.real, aaa.imag, c=c)

    # if len(diff_d) < len(diff_f):
    #     i = len(diff_f)
    #     while i > len(diff_d):
    #         diff_d.append(0)
    #         i = i - 1
    # else:
    #     i = len(diff_f)
    #     while i < len(diff_d):
    #         diff_f.append(0)
    #         i = i + 1
    # print("len_f: ", len(diff_f), " len_d", len(diff_d))
    scenes = list(range(0, len(diff_d)))
    plt.scatter(scenes, diff_d, marker='x', color="red", linewidths=1, label="desert")
    scenes = list(range(0, len(diff_f)))
    plt.scatter(scenes, diff_f, marker='x', color="green", linewidths=1, label="forest")
    scenes = list(range(0, len(diff_s)))
    plt.scatter(scenes, diff_s, marker='x', color="black", linewidths=1, label="snow")
    plt.xlabel('Scene')
    plt.ylabel('Difference in Days')
    plt.title('Date Difference Scatter Plot')
    plt.legend()
    plt.savefig('date_diff_plot.png')

def plot_training_validation_loss(training_loss, validation_loss):
    epoch = [i for i in range(len(training_loss))]

    l1, = plt.plot(epoch, training_loss, color = "red")
    l2, = plt.plot(epoch, validation_loss, color = "blue")
    plt.legend((l1,l2), ["Training Loss", "Validation Loss"])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss.')
    plt.savefig('training_validation_loss.png')

