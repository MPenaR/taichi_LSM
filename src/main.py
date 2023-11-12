from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps
import numpy as np
import numpy.typing as npt
import taichi as ti
from scipy.special import j0, y0


from databases import FresnelDatabase


database= FresnelDatabase()


#database.load(U)

database.load(Path("../Fresnel_Data") / "uTM_shaped.txt")


FF = database.FarField()

M = 400
N = 400

res = (M,N)

Lx = 0.1
Ly = Lx * N / M

x_R, y_R = database.r_R
N_E = database.N_E
N_F = database.N_F
kappa = database.kappa

u, s, _ = np.linalg.svd(FF, full_matrices=True)


x = np.linspace(-Lx,Lx,M)
y = np.linspace(-Ly,Ly,N)
X, Y = np.meshgrid(x,y,indexing='ij')

r_x = np.subtract.outer(X, x_R)
r_y = np.subtract.outer(Y, y_R)


# HERE IS THE BOTLE NECK
def h0(x : npt.NDArray[np.float64]) ->  npt.NDArray[np.complex128] : 
    return j0(x) + 1j*y0(x)

b =  1j/4 * h0(np.multiply.outer( kappa, np.sqrt( r_x**2 + r_y**2) ))

#t1 = perf_counter()
# b_R = -1/4 * y0(np.multiply.outer( kappa, np.sqrt( r_x**2 + r_y**2) ))
# b_I =  1/4 * j0(np.multiply.outer( kappa, np.sqrt( r_x**2 + r_y**2) ))

#t2  = perf_counter()
#print(f'{t2 - t1 = }')


ubub = np.zeros((*res, N_F, N_E), dtype=np.float64)

for k in range(len(kappa)):
    for n in range(N_E):
        ubub[:,:,k,n] = np.abs(np.tensordot(np.conj(u[k,:,n]), b[k,:,:,:], axes=( (0), (-1))))**2

ggui = True
arch = ti.vulkan if ggui else ti.gpu
ti.init(arch=arch, unrolling_limit=0)

UBUB = ti.ndarray( dtype= ti.types.matrix(n=N_F, m=N_E, dtype=ti.f64), shape = res)

UBUB.from_numpy(ubub)

big_res = (4*N, 2*N)
window = ti.ui.Window(f'single frequency', res=big_res)
canvas = window.get_canvas()
np_pix = np.zeros(dtype=np.uint8,shape=( N_F, *res))
pixels = ti.field(dtype=ti.f32, shape = res)
big_pix = ti.field(dtype=ti.f32, shape = big_res)
np_big_pix = np.zeros(dtype=np.uint8,shape=big_res)
ind = ti.field(dtype=ti.f32, shape = res)

S = ti.ndarray(dtype=ti.f64, shape=(N_F, N_E))
S.from_numpy(s)

@ti.kernel
def mono_LSM( UBUB : ti.types.ndarray(), S : ti.types.ndarray(), a : ti.f32, l : ti.f32, k : int ):
    for i, j in ind:
        for n in ti.static(range(N_E)):
            ind[i,j] += (S[k,n]/(S[k,n]**2 + a))**2*UBUB[i,j][k,n]
        ind[i,j] = 1. / ti.sqrt(ind[i,j])
        pixels[i,j] = ti.u8(255) if ind[i,j] > l else ti.u8(0)

# cmap = colormaps["viridis"]
#Visualization
while window.running:
    mouse_x, mouse_y = window.get_cursor_pos()
    l = 0.01*(mouse_y) + 0.988
    a = 1E-1*mouse_x
    print(f'{l= : .6f} {a= : .6f}', end='\r')
    for k in range(len(kappa)):
        mono_LSM(UBUB, S, a, l, k )
        np_pix[k,:,:] = pixels.to_numpy()
    
    np_big_pix[:, N: ] = np.fliplr(np.concatenate( [ np_pix[k,:,:] for k in range(4)   ], axis=0 ) )
    np_big_pix[:,  :N] = np.fliplr(np.concatenate( [ np_pix[k,:,:] for k in range(4,8) ], axis=0 ) )
    big_pix.from_numpy(np_big_pix)
    canvas.set_image(big_pix)
    window.show()
print('')






# b_R, b_I = compute_b( X, Y, x_R, y_R, kappa)

# b = ti.Narray(ti.types.vector(n=2,dtype=ti.f32), shape=(N_F, *res))


# @ti.kernel
# def compute_ub2( b : ti.ndarray):



# @ti.kernel
# def compute_ub_squared():
#     for i, j in ub2:
#         r_x = x_R - x[i,j]
#         r_y = y_R - y[i,j]
#         b = 1j/4 * hankel1(0, np.multiply.outer( k, np.sqrt( r_x**2 + r_y**2) ))
#         for f in ti.static(range(N_F)):
#             ubub = np.abs(np.dot(u[f,:,:N_E], b[f,:]))**2
#             for r in ti.static(range(N_E)):
#                 ub2[i,j][k,r] = ubub[r]

# compute_ub_squared()

# print(f'{b.shape = }') # (8, 1200, 800, 72)





# vec_N_E = dtype=ti.types.vector( n = N_E, dtype = ti.f32 )

# ubub = ti.ndarray( vec_N_E, shape = res )
# ubub.from_numpy( np.abs(np.transpose(np.tensordot(np.conj(u[:,:N_E]), b, axes=(0,2)), (1,2,0)))**2)


# cmap = colormaps["viridis"]


# ind = ti.field(dtype=ti.f32, shape = res )
# pixels = ti.field(dtype=ti.u8, shape = res )

# @ti.kernel
# def LSM( ubub : ti.types.ndarray(vec_N_E, ndim=2), a : ti.f32, l : ti.f32 ):
#     for i, j in ind:
#         for n in ti.static(range(N_E)):
#             ind[i,j] += (s[n]/(s[n]**2 + a))**2*ubub[i,j][n]
#         ind[i,j] = 1. / ti.sqrt(ind[i,j])
#         #ind[i,j] = 1. if ind[i,j] > l else 0.
#         pixels[i,j] = ti.u8(255) if ind[i,j] > l else ti.u8(0)






# window = ti.ui.Window(f'LSM at f = { database.frequencies[f_ID]/1E9 : .2f} GHz', res=res)
# canvas = window.get_canvas()


# # result_dir = "./results_u8"
# # video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)


# #Visualization
# while window.running:
#     mouse_x, mouse_y = window.get_cursor_pos()
#     l = 0.01*(mouse_y) + 0.988
#     a = 1E-1*mouse_x
#     print(f'{l= : .6f} {a= : .6f}', end='\r')
#     LSM(ubub, a, l )
#     canvas.set_image(pixels)
#     window.show()
#     #video_manager.write_frame(pixels)
# print('')

# #video_manager.make_video(gif=True, mp4=True)

