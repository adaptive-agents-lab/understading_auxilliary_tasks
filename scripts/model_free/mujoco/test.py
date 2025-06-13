import glob
import numpy as np
import sys

path = sys.argv[1]

x = sorted(glob.glob(path + "checkpoint/replay*"))

for f in x:
  try:
    np.load(f)
  except Exception as e:
    print(f)
    print(e)
print(x[-1])
