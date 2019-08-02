
import dolfin
import logging

from dolfin import Constant
from dolfin import assemble
from dolfin import grad
from dolfin import dot
from dolfin import dx

from . import config
logger = config.logger


class GradientFilter:
    def __init__(self, V):
        '''
        Parameters
        ----------
        V : dolfin.FunctionSpace
            Function space.

        '''

        if not isinstance(V, dolfin.FunctionSpace):
            raise TypeError('Parameter `V` must be of type `dolfin.FunctionSpace`.')

        v0 = dolfin.TestFunction(V)
        v1 = dolfin.TrialFunction(V)

        A = assemble(v0*v1*dx) # Mass-like matrix
        self.solver = dolfin.LUSolver(A, "mumps")
        self.solver.parameters["symmetric"] = True

    def apply(self, fn):
        x = fn.vector(); b = x.copy()
        return self.solver.solve(x, b)


class DiffusionFilter:
    def __init__(self, V, kappa):
        '''
        Parameters
        ----------
        V : dolfin.FunctionSpace
            Function space.
        kappa : float
            Filter diffusivity constant.

        '''

        if not isinstance(V, dolfin.FunctionSpace):
            raise TypeError('Parameter `V` must be of type `dolfin.FunctionSpace`.')

        v = dolfin.TestFunction(V)
        f = dolfin.TrialFunction(V)

        x = V.mesh().coordinates()
        l = (x.max(0)-x.min(0)).min()
        k = Constant(float(kappa)*l**2)

        a_M = f*v*dx
        a_D = k*dot(grad(f), grad(v))*dx

        A = assemble(a_M + a_D)
        self._M = assemble(a_M)

        self.solver = dolfin.LUSolver(A, "mumps")
        self.solver.parameters["symmetric"] = True

    def apply(self, fn):
        x = fn.vector(); b = self._M*x
        return self.solver.solve(x, b)