import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes = ['Irrigation Area', 'Snow Cap Mountains', 'Urban Area']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()


def plot_mertics(actual, predicted, title):
    unique_labels = np.unique(actual + predicted)

    confusion = confusion_matrix(actual, predicted, labels=unique_labels)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(confusion, classes=unique_labels, title=title)
    plt.savefig('confusion_matrix_req.png')
