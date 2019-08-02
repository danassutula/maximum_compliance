
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
