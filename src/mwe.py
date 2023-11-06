import numpy as np
import taichi as ti 

ti.init()
dim = 3
N = 100
res = (N,N)

ti_arr = ti.ndarray( dtype=ti.types.vector(n=dim, dtype=ti.f32), shape=res)
arr = np.ones(dtype=np.float32, shape=( *res, dim))
ti_arr.from_numpy(arr)


arr_sum = ti.field( dtype=ti.f32, shape=res)

@ti.kernel
def perform_sum( ti_arr : ti.types.ndarray( dtype=ti.types.vector(n=dim, dtype=ti.f32), ndim=2)  ):
    for i, j in arr_sum:
        arr_sum[i,j] = ti_arr[i,j][0] + ti_arr[i,j][1] + ti_arr[i,j][2]

perform_sum(ti_arr)
