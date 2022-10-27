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
    data[i] = list(map(lambda x: float(x), ln.strip().split(',')))
  return (header, data) if headed else data

def graph(data, title):
  plt.title(title)
  plt.plot(data[:], 'r') 
  plt.show()

def main():
  if len(sys.argv) < 3:
    print('Usage: ./graph.py <path-to-losses-file> <title>')
    exit(1)
  data = np.array(read_csv_file(sys.argv[1], headed=False))
  graph(data, sys.argv[2])

if __name__ == "__main__":
  main()
