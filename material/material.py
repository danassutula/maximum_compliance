'''
Material models in terms of the energy density that take the displacement field
and the material parameters as arguments.

'''

import logging
import dolfin

from dolfin import Constant
from dolfin import Function
from dolfin import Identity
from dolfin import variable

from dolfin import det
from dolfin import diff
from dolfin import dot
from dolfin import grad
from dolfin import inner
from dolfin import inv
from dolfin import ln
from dolfin import sym
from dolfin import tr

logger = logging.getLogger()


class DeformationMeasures:
    def __init__(self, u):
        '''Deformation measures.'''

        self.d = d = len(u)
        self.I = I = Identity(d)
        self.F = F = variable(I + grad(u))

        self.C = C = F.T*F
        self.E = 0.5*(C-I)
        self.J = det(F)

        self.I1 = tr(C)
        self.I2 = 0.5*(tr(C)**2 - tr(C*C))
        self.I3 = det(C)


class AbstractModel:
    def __init__(self, material_parameters, u=None):
        '''Base class for deriving a specific material model.'''

        if isinstance(material_parameters, (list, tuple)):
            self.material_parameters = material_parameters
            self.deformation_measures = None
            self._to_return_iterable = True
        else:
            self.material_parameters = (material_parameters,)
            self.deformation_measures = None
            self._to_return_iterable = False

        if not all(isinstance(m, dict) for m in self.material_parameters):
            raise TypeError('`material_parameters` must be a `dict` '
                            'or a `list` or `tuple` of `dict`s.')

        self.psi = [] # Strain energy density
        self.pk1 = [] # First Piola-Kirchhoff stress
        self.pk2 = [] # Second Piola-Kirchhoff stress

        if u is not None:
            self.initialize_with_field(u)

    def initialize_with_field(self, u):
        '''To be extended by derived class.'''

        if self.is_initialized():
            logger.info('Re-finalizing material model.')

            self.psi.clear()
            self.pk1.clear()
            self.pk2.clear()

        self.deformation_measures = DeformationMeasures(u)

    def is_initialized(self):
        '''Check if initialized (by derived class)'''
        return self.deformation_measures and \
            self.psi and self.pk1 and self.pk2

    def strain_energy_density(self):
        '''Material model strain energy density.'''
        if not self.is_initialized(): raise RuntimeError('Not initialized.')
        return self.psi if self._to_return_iterable else self.psi[0]

    def stress_measure_pk1(self):
        '''Material model First Piola-Kirchhoff stress measure.'''
        if not self.is_initialized(): raise RuntimeError('Not initialized.')
        return self.pk1 if self._to_return_iterable else self.pk1[0]

    def stress_measure_pk2(self):
        '''Material model Second Piola-Kirchhoff stress measure.'''
        if not self.is_initialized(): raise RuntimeError('Not initialized.')
        return self.pk2 if self._to_return_iterable else self.pk2[0]


class LinearElasticModel(AbstractModel):
    def initialize_with_field(self, u):
        super().initialize_with_field(u)

        I = Identity(len(u))
        eps = sym(grad(u))

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

            sig = 2*mu*eps + lm*tr(eps)*I
            psi = 0.5 * inner(sig, eps)

            self.psi.append(psi)
            self.pk1.append(sig)
            self.pk2.append(sig)


class NeoHookeanModel(AbstractModel):
    def initialize_with_field(self, u):
        super().initialize_with_field(u)

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

            self.psi.append(psi)
            self.pk1.append(pk1)
            self.pk2.append(pk2)
