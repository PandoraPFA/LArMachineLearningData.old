import numpy as np

def main():

  npixels=128

  data = np.loadtxt('data.txt')
  data = np.reshape(data, (data.shape[0]/npixels,npixels,npixels))
  for i in range(0,data.shape[0]):
    data[i] = np.flipud(data[i])
  data = np.reshape(data, (data.shape[0],npixels,npixels,1))
  np.save('data', data)

  labels = np.loadtxt('labels.txt')
  np.save('labels', labels)

if __name__ == "__main__":

  main()
