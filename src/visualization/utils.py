import matplotlib.pyplot as plt
# Adding comment
import numpy as np

def plot_loss(epochs, losses):
  plt.plot(epochs, losses)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss as a function of epochs')
  plt.savefig("/mnt/c/Users/Lasse/Desktop/DTU/7. semester/MLOps/MLOPS/reports/figures/train_loss.png")
  plt.show()
