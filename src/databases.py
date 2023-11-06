"""Module containing the different loaders"""

from typing import Protocol
from abc import abstractmethod
from pathlib import Path


import numpy as np
import numpy.typing as npt
from scipy.constants import c

class Database(Protocol):
    """General protocol for databases."""
    N_E : int
    N_R : int
    N_M : int 
    N_F : int 
    E_sc  : npt.NDArray[np.complex64]
    U_inc : npt.NDArray[np.complex64]
    r_ID  : npt.NDArray[np.int32]
    frequencies : npt.NDArray[np.float32]
    R_E : float
    R_R : float 

    @abstractmethod
    def load(self, file_path : Path):
        """Loads the file given in 'file_path'"""
        raise NotImplementedError

    @property
    def kappa(self) -> npt.NDArray[np.float32]:
        """wavelenghts"""
        return 2*np.pi*self.frequencies / c
    
    @property
    def theta_E(self) -> npt.NDArray[np.float64]:
        """Angular position of the emitters.s"""
        return np.linspace( 0, 2*np.pi, self.N_E, endpoint=False)
    
    @property
    def r_E(self) -> tuple:
        """Cartesian position of the emitters."""
        return ( self.R_E*np.cos(self.theta_E), self.R_E*np.sin(self.theta_E) )

    @property
    def theta_R(self) -> npt.NDArray[np.float64]:
        """Angular position of the receivers."""
        return np.linspace( 0, 2*np.pi, self.N_R, endpoint=False)
    @property
    def r_R(self) -> tuple:
        """"Cartesian position of the receivers."""
        return ( self.R_R*np.cos(self.theta_R), self.R_R*np.sin(self.theta_R) )


    def FarField( self, zero_fill = True) -> npt.NDArray[np.complex128] :
        """Returns the far field matrix. if `zero_field` then
        the missing data is completed with zeroes.
        """ 
        FF = np.zeros( (self.N_F, self.N_R, self.N_E), dtype=np.complex128)
        if zero_fill: 
            for e in range(self.N_E):
                FF[ :, self.r_ID[:,e], e ] = self.E_sc[:,:,e]
        return FF

filepath = Path("../Fresnel_Data") / "uTM_shaped.txt"
# filepath = Path("../Fresnel_Data") / "twodielTM_8f.txt"
# filepath = Path("../Fresnel_Data") / "rectTM_dece.txt"
# filepath = Path("../Fresnel_Data") / "dielTM_dec8f.txt"

 

class FresnelDatabase(Database):
    """Institut Fresnel 2D database"""
    
    def __init__(self):
        self.N_E = 36
        self.N_M = 49
        self.N_R = 72
        self.R_E = 0.72
        self.R_R = 0.76

    def load(self, file_path : Path):
        """Loads the file given in 'file_path'"""
        
        with open(file=file_path, mode='r') as datafile:
                for _ in range(4):
                    next(datafile)
                line = datafile.readline()
                self.N_F = int(line.split(":")[1].split("(")[0])
                self.frequencies = np.array([ int(f)*1.0E9 for f in line.split(":")[1].split("(")[1].split("G")[0].split(",") ])

        shape = (self.N_F, self.N_M, self.N_E)
        data = np.loadtxt(file_path, skiprows=10)
        self.r_ID  = (data[:,1].reshape( shape, order='F' ) - 1).astype(int)[0,:,:]
        E_tot = ( data[:,3] - 1j*data[:,4] ).reshape( shape, order='F' )
        self.U_inc  = ( data[:,5] - 1j*data[:,6] ).reshape( shape, order='F' )
        self.E_sc = E_tot - self.U_inc


class ManitobaDatabase(Database):
    """Institut Fresnel 2D database"""
    
    def __init__(self):
        self.N_E = 24
        self.N_M = 23
        self.N_R = 24
        self.R_E = 0.12
        self.R_R = 0.12

    def load(self, file_path : Path):
        """Loads the file given in 'file_path'"""
        
        with open(file=file_path, mode='r') as datafile:
                for _ in range(4):
                    next(datafile)
                line = datafile.readline()
                self.N_F = int(line.split(":")[1].split("(")[0])
                self.frequencies = np.array([ int(f)*1.0E9 for f in line.split(":")[1].split("(")[1].split("G")[0].split(",") ])

        shape = (self.N_F, self.N_M, self.N_E)
        data = np.loadtxt(file_path, skiprows=10)
        self.r_ID  = (data[:,1].reshape( shape, order='F' ) - 1).astype(int)[0,:,:]
        E_tot = ( data[:,3] - 1j*data[:,4] ).reshape( shape, order='F' )
        self.U_inc  = ( data[:,5] - 1j*data[:,6] ).reshape( shape, order='F' )
        self.E_sc = E_tot - self.U_inc





if __name__ == '__main__':
    def takes_a_database( L : Database ):
        pass 
    L = FresnelDatabase()
