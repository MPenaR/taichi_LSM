import taichi as ti
import taichi.math as tm




@ti.func
def JP( x ):
    A =(-6.068350350393235E-8, 
         6.388945720783375E-6, 
        -3.969646342510940E-4,
         1.332913422519003E-2,
        -1.729150680240724E-1)
    s = 0.
    for n in range(len(A)):
        s += A[-1-n]*x**n
    return s

@ti.func
def MO( x: float ):
    A =(-6.838999669318810E-2,
         1.864949361379502E-1,
        -2.145007480346739E-1,
         1.197549369473540E-1,
        -3.560281861530129E-3,
        -4.969382655296620E-2,
        -3.355424622293709E-6,
         7.978845717621440E-1)
    s = 0.
    for n in range(len(A)):
        s += A[-1-n]*x**n
    return s

@ti.func
def PH( x: float ):
    A =( 3.242077816988247E+1,
        -3.630592630518434E+1,
         1.756221482109099E+1,
        -4.974978466280903E+0,
         1.001973420681837E+0,
        -1.939906941791308E-1,
         6.490598792654666E-2,
        -1.249992184872738E-1)
    s = 0.
    for n in range(len(A)):
        s += A[-1-n]*x**n
    return s

@ti.func
def j0( x ):
    r1 = 5.78318596294678452118

    if x < 0:
        x = -x 

    if x <= 2. :
        z = x*x
        if x < 1E-3:
            y = 1. - 0.25*z
        else: 
            y = ( z - r1 ) * JP(z)
    else:
        q = 1. / x
        w = tm.sqrt(q)

        p = w * MO(q)
        w = q*q
        xn = q * PH(w) - tm.pi / 4
        y = p * tm.cos( xn + x )
    return y 


if __name__ == "__main__":
    import numpy as np

    
    ti.init(arch=ti.gpu)
    N = 8
    res = (N, N)
    J = ti.field(dtype=ti.f32, shape = (N))
    k = 1
    x = np.linspace(0,5,N,dtype=np.float32)
    X = ti.ndarray(dtype=ti.f32, shape=(N))
    
    X.from_numpy(x)

    @ti.kernel
    def use_j0( X : ti.types.ndarray()):
        for i in J:
            J[i] = j0(X[i])
    
    use_j0(X)