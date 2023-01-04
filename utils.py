import matplotlib.pyplot as plt

def plot_loss(epochs, losses):
  plt.plot(epochs, losses)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss as a function of epochs')
  plt.savefig("train_loss.png")
  plt.show()