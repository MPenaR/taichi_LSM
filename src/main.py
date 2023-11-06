from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps
import numpy as np
import taichi as ti
from scipy.special import hankel1

from databases import FresnelDatabase


database= FresnelDatabase()


#database.load(U)

database.load(Path("../Fresnel_Data") / "uTM_shaped.txt")


FF = database.FarField()

M = 1200
N = 800

res = (M,N)

Lx = 0.3
Ly = Lx * N / M 


x_R, y_R = database.r_R
N_E = database.N_E
kappa = database.kappa

f_ID = 7

k = kappa[f_ID]



x = np.linspace(-Lx,Lx,M)
y = np.linspace(-Ly,Ly,N)
X, Y = np.meshgrid(x,y,indexing='ij')


r_x = np.subtract.outer(X, x_R)
r_y = np.subtract.outer(Y, y_R)

b = 1j/4 * hankel1(0,k*np.sqrt( r_x**2 + r_y**2))

u, s, vh = np.linalg.svd(FF[f_ID,:,:], full_matrices=True)

ggui = True

arch = ti.vulkan if ggui else ti.gpu

ti.init(arch=arch, unrolling_limit=0)

vec_N_E = dtype=ti.types.vector( n = N_E, dtype = ti.f32 )

ubub = ti.ndarray( vec_N_E, shape = res )
ubub.from_numpy( np.abs(np.transpose(np.tensordot(np.conj(u[:,:N_E]), b, axes=(0,2)), (1,2,0)))**2)


# cmap = colormaps["viridis"]


ind = ti.field(dtype=ti.f32, shape = res )
pixels = ti.field(dtype=ti.u8, shape = res )

@ti.kernel
def LSM( ubub : ti.types.ndarray(vec_N_E, ndim=2), a : ti.f32, l : ti.f32 ):
    for i, j in ind:
        for n in ti.static(range(N_E)):
            ind[i,j] += (s[n]/(s[n]**2 + a))**2*ubub[i,j][n]
        ind[i,j] = 1. / ti.sqrt(ind[i,j])
        #ind[i,j] = 1. if ind[i,j] > l else 0.
        pixels[i,j] = ti.u8(255) if ind[i,j] > l else ti.u8(0)






window = ti.ui.Window(f'LSM at f = { database.frequencies[f_ID]/1E9 : .2f} GHz', res=res)
canvas = window.get_canvas()


result_dir = "./results_u8"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)



while window.running:
    mouse_x, mouse_y = window.get_cursor_pos()
    l = 0.01*(mouse_y) + 0.988
    a = 1E-1*mouse_x
    print(f'{l= : .6f} {a= : .6f}', end='\r')
    LSM(ubub, a, l )
    canvas.set_image(pixels)
    window.show()
    video_manager.write_frame(pixels)
print('')

video_manager.make_video(gif=True, mp4=True)
