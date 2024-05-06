from motifs import *
from utils import *
import matplotlib.pyplot as plt


func_mat = load_matrix('func_matrix.npy')
plt.imshow(func_mat)
plt.colorbar()
plt.show()
save_motifs(func_mat, 'natural_scene')
