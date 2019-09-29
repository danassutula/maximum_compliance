

class PeriodicBoundaries(dolfin.SubDomain):
    '''Top boundary is slave boundary wrt bottom (master) boundary.
    Right boundary is slave boundary wrt left (master) boundary.'''

    def __init__(self, mesh):
        super().__init__() # Important!

        x0, y0 = mesh.coordinates().min(0)
        x1, y1 = mesh.coordinates().max(0)

        self.L = x1 - x0
        self.H = y1 - y0

        self.x_lhs = x0 + EPS*self.L
        self.y_bot = y0 + EPS*self.H

        self.x_rhs = x1 - EPS*self.L
        self.y_top = y1 - EPS*self.H

    def inside(self, x, on_boundary):
        '''Check if `x` is on the slave boundary'''
        return on_boundary and \
            ((x[0] > self.x_rhs and x[1] > self.y_bot) or
             (x[0] > self.x_lhs and x[1] > self.y_top))

    def map(self, x, y):
        '''Map master point `x` to slave point `y`.'''

        y[0] = x[0]
        y[1] = x[1]

        if x[0] < self.x_lhs and x[1] > self.y_top:
            pass

        elif x[0] > self.x_rhs and x[1] < self.y_bot:
            pass

        elif x[0] < self.x_lhs:
            y[0] += self.L

        elif x[1] < self.y_bot:
            y[1] += self.H
            


normL1_dp = self._domain_integral_dof_weights.dot(abs_dp_arr)


self._domain_integral_dof_weights = \
    assemble(dolfin.TestFunction(self._V_p)*dx).get_local()

        # collision_distance_hard : float
        #     Minimum distance between local phasefields excluding blending branch.

self._d_collision_hard = collision_distance_hard


        self._collision_blending_length = \
            self._d_collision - self._d_collision_hard


        self._d_collision_hard = None
        self._collision_blending_length = None



    def _apply_collision_prevention(self, p_arr):

        # NOTE: The "distance" needs to be defined as the second smallest distance
        # because the smallest value just is a reference to a particular phasefield.

        ind = np.flatnonzero((self._d_arr_locals < self._d_collision).sum(0) > 1) # Avoid sorting
        mask_hard = (self._d_arr_locals[:,ind] < self._d_collision_hard).sum(0) > 1 # Avoid sorting

        ind_hard = ind[mask_hard]
        ind_soft = ind[~mask_hard]

        s_arr = (np.sort(self._d_arr_locals[:,ind_soft], 0)[1] # 2nd
            - self._d_collision_hard) / self._collision_blending_length

        assert np.all(s_arr < 1+EPS)
        assert np.all(s_arr > -EPS)

        p_arr[ind_soft] *= self._collision_smoothing_weight(s_arr)
        p_arr[ind_hard] = 0.0


        if not (0.0 <= collision_distance_hard <= collision_distance):
            raise ValueError('0 < collision_distance_hard < collision_distance')

                    # Assign previous phasefield solution
                    # NOTE: `p_arr_prv` satisfies constraints

                    self._assign_phasefield_values(p_arr_prv)
                    self._solve_phasefield_distances()

                    _, b = self._solve_equilibrium_problem()

                    if not b:
                        logger.error('Displacement problem could not be solved')


            if not b and n <= 2: # Trigger for redoing last solution

                phasefield_volume_fraction_i -= \
                    phasefield_volume_fraction_stepsize

                phasefield_volume_fractions.pop(-1)
                potential_vs_phasefield.pop(-1)

                p_mean.assign(phasefield_volume_fraction_i)

                n, b, potential_vs_iteration_i = optimizer.optimize(
                    phasefield_stepsize_final, penalty_weight, collision_distance,
                    collision_distance_hard, convergence_tolerance_final)

                potential_vs_iteration.extend(potential_vs_iteration_i)
                potential_vs_phasefield[-1] = potential_vs_iteration_i[-1]

                break

        # if x is not None:
        #    self._solve_initdist_problem()
        #    x[:] = self._d_vec.get_local()
        # else:
        #     self._solve_initdist_problem()
        #     x = self._d_vec.get_local()
        #
        # return x

        # if x is not None:
        #     self._d_vec[:] = x
        #     try:
        #         self._solve_distance_problem()
        #     except RuntimeError:
        #         self._solve_initdist_problem()
        #         self._solve_distance_problem()
        #     x[:] = self._d_vec.get_local()
        # else:
        #     try:
        #         self._solve_distance_problem()
        #     except RuntimeError:
        #         self._solve_initdist_problem()
        #         self._solve_distance_problem()
        #     x = self._d_vec.get_local()
        #
        # return x

mask = (self._d_arr_locals < collision_distance).sum(0) > 1
dist = np.sort(self._d_arr_locals[:,mask], 0)[:2].sum(0)*0.5
weight = (dist / collision_distance)**2

p_arr[mask] *= weight
