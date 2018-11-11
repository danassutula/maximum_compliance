'''
Material models in terms of the energy density that take the displacement field
and the material parameters as arguments.

'''

import dolfin

from dolfin import Constant
from dolfin import Function
from dolfin import Identity
from dolfin import variable

from dolfin import det
from dolfin import diff
from dolfin import dot
from dolfin import grad
from dolfin import inv
from dolfin import ln
from dolfin import tr


class DeformationMeasures:
    def __init__(self, u):
        '''Deformation measures.'''

        if not isinstance(u, Function):
            raise TypeError('Parameter `u` must be a `dolfin.Function`.')

        self.d = d = len(u)
        self.I = I = Identity(d)
        self.F = F = variable(I + grad(u))

        self.C = C = F.T*F
        self.E = 0.5*(C-I)
        self.J = det(F)

        self.I1 = tr(C)
        self.I2 = 0.5*(tr(C)**2 - tr(C*C))
        self.I3 = det(C)


class MaterialModelBase:
    def __init__(self, u, material_parameters):
        '''Base class for deriving a material model.'''

        if isinstance(material_parameters, (list,tuple)):
            self.material_parameters = material_parameters
            self._to_return_iterable = True
        else:
            self.material_parameters = (material_parameters,)
            self._to_return_iterable = False

        if not all(isinstance(m, dict) for m in self.material_parameters):
            raise TypeError('`material_parameters` must be a `dict` or '
                            'a `list` or `tuple` of `dict`s.')

        if not all(isinstance(m_i, Constant)
                for m in self.material_parameters for m_i in m.values()):
            raise TypeError('`material_parameters` must contain values '
                            'of type `dolfin.Constant`.')

        self.deformation_measures = DeformationMeasures(u)

    def strain_energy_density(self):
        raise NotImplementedError('Require override.')

    def stress_measure_pk1(self):
        raise NotImplementedError('Require override.')

    def stress_measure_pk2(self):
        raise NotImplementedError('Require override.')


class NeoHookeanModel(MaterialModelBase):
    def __init__(self, u, material_parameters):
        super().__init__(u, material_parameters)
        # -> self.material_parameters,
        # -> self._to_return_iterable

        self._psi = [] # Strain energy density
        self._pk1 = [] # First Piola-Kirchhoff stress
        self._pk2 = [] # Second Piola-Kirchhoff stress

        d  = self.deformation_measures.d
        F  = self.deformation_measures.F
        J  = self.deformation_measures.J
        I1 = self.deformation_measures.I1
        # I2 = self.deformation_measures.I2
        # I3 = self.deformation_measures.I3

        for m in self.material_parameters:

            E  = m.get('E',  None)
            nu = m.get('nu', None)

            mu = m.get('mu', None)
            lm = m.get('lm', None)

            if mu is None:
                if E is None or nu is None:
                    raise RuntimeError('Material model requires parameter "mu"; '
                                       'otherwise, require parameters "E" and "nu".')

                mu = E/(2*(1 + nu))

            if lm is None:
                if E is None or nu is None:
                    raise RuntimeError('Material model requires parameter "lm"; '
                                       'otherwise, require parameters "E" and "nu".')

                lm = E*nu/((1 + nu)*(1 - 2*nu))

            psi = (mu/2)*(I1 - d - 2*ln(J)) + (lm/2)*ln(J)**2
            pk1 = diff(psi, F)
            pk2 = dot(inv(F), pk1)

            self._psi.append(psi)
            self._pk1.append(pk1)
            self._pk2.append(pk2)

    def strain_energy_density(self):
        '''Neo-Hookean material strain energy density.'''
        return self._psi if self._to_return_iterable else self._psi[0]

    def stress_measure_pk1(self):
        '''Neo-Hookean material PK1 stress measure.'''
        return self._pk1 if self._to_return_iterable else self._pk1[0]

    def stress_measure_pk2(self):
        '''Neo-Hookean material PK2 stress measure.'''
        return self._pk2 if self._to_return_iterable else self._pk2[0]
