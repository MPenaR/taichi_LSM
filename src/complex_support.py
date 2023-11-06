import taichi as ti


@ti.dataclass
class ComplexScalarField:
    z : ti.types.vector[2]
    def __init__(self, val : complex ):
            self.z[0], self.z[1] = val.real, val.imag
    
    def prod(self, other : ComplexScalarField) -> ComplexScalarField:
          return ComplexScalarField( self.z * other.z )
         

          

if __name__ == "__main__":
      z1 = ComplexScalarField(1 + 1j)
      z2 = ComplexScalarField(1 - 1j)
      print( (z1.prod(z2)).z)