
import dolfin

from dolfin import Constant
from dolfin import assemble
from dolfin import grad
from dolfin import dot
from dolfin import dx


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


def apply_diffusive_smoothing(fn, kappa):

    if isinstance(fn, (list, tuple)):
        return_as_sequence = True
    else:
        return_as_sequence = False
        fn = (fn,)

    if not all(isinstance(fn_i, dolfin.Function) for fn_i in fn):
        raise TypeError('Parameter `fn` must either be a `dolfin.Function` '
                        'or a sequence (list, tuple) of `dolfin.Function`s.')

    diffusion_filter = DiffusionFilter(fn[0].function_space(), kappa)

    for fn_i in fn:
        diffusion_filter.apply(fn_i)

    return fn if return_as_sequence else fn[0]


def trimoff_function_values(fn, lower=0.0, upper=1.0):

    if isinstance(fn, (list, tuple)):
        return_as_sequence = True
    else:
        return_as_sequence = False
        fn = (fn,)

    if not all(isinstance(fn_i, dolfin.Function) for fn_i in fn):
        raise TypeError('Parameter `fn` must either be a `dolfin.Function` '
                        'or a sequence (list, tuple) of `dolfin.Function`s.')

    for fn_i in fn:

        x = fn_i.vector().get_local()

        x[x < lower] = lower
        x[x > upper] = upper

        fn_i.vector().set_local(x)

    return fn if return_as_sequence else fn[0]


def rescale_function_values(fn, lower=0.0, upper=1.0):

    if isinstance(fn, (list, tuple)):
        return_as_sequence = True
    else:
        return_as_sequence = False
        fn = (fn,)

    if not all(isinstance(fn_i, dolfin.Function) for fn_i in fn):
        raise TypeError('Parameter `fn` must either be a `dolfin.Function` '
                        'or a sequence (list, tuple) of `dolfin.Function`s.')

    for fn_i in fn:

        x = fn_i.vector().get_local()

        xmin = x.min()
        xmax = x.max()

        x -= xmin
        x *= (upper-lower) / (xmax-xmin)
        x += lower

        fn_i.vector().set_local(x)

    return fn if return_as_sequence else fn[0]
