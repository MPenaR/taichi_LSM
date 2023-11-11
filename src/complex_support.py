import taichi as ti
import numpy as np

@ti.data_oriented
class ComplexScalarField:
    def __init__(self, val : complex ):
        self.val = ti.Vector( arr= (val.real, val.imag ), dt = ti.f32 )

    @ti.kernel
    def prod(self : ComplexScalarField, other : ComplexScalarField) -> ComplexScalarField:
        return ComplexScalarField( self.z * other.z )
         

          

if __name__ == "__main__":
      z1 = ComplexScalarField(1 + 1j)
      z2 = ComplexScalarField(1 - 1j)
      print( (prod(z1, z2)).z)