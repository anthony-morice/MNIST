#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt

def read_csv_file(file_path, headed = True):
  # read data from file
  fh = open(file_path, "r")
  header = fh.readline().strip().split(',') if headed else None
  data = fh.readlines()
  fh.close()
  # remove newline and split each line using comma as delimiter
  for i, ln in enumerate(data):
    data[i] = list(map(lambda x: int(x), ln.strip().split(',')))
  return (header, data) if headed else data

def confusion_matrix(m, accuracy, class_labels, title='confusion matrix'):
  plt.title("{}, accuracy = {:.3f}".format(title, accuracy))
  plt.imshow(m)
  ax, fig = plt.gca(), plt.gcf()
  plt.xticks(np.arange(len(class_labels)), class_labels)
  plt.yticks(np.arange(len(class_labels)), class_labels)
  ax.set_xticks(np.arange(len(class_labels) + 1) - .5, minor=True)
  ax.set_yticks(np.arange(len(class_labels) + 1) - .5, minor=True)
  ax.tick_params(which="minor", bottom=False, left=False)
  plt.show()

def main():
  if len(sys.argv) < 2:
    print('Usage: ./confusion.py <path-to-classification-results>')
    exit(1)
  data = np.array(read_csv_file(sys.argv[1], headed=False))
  class_labels = [str(i) for i in range(10)]
  acc = 0
  confusion = np.zeros((10, 10))
  for i in range(data.shape[0]):
    pred, label = data[i]
    confusion[pred, label] += 1
    if pred == label:
      acc += 1
  accuracy = acc / data.shape[0]
  for i in range(10):
    confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])
  confusion_matrix(confusion, accuracy, class_labels, 'MLP Confusion Matrix')
  
if __name__ == "__main__":
  main()
