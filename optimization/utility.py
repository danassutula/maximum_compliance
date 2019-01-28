
def update_lhs_parameters_from_rhs(lhs, rhs):
    '''Update parameter values of dict-like `lhs` with those from `rhs`.'''

    for k in rhs.keys():

        if k not in lhs.keys():
            raise KeyError(k) # NOTE: `k` in `rhs.keys()` is not in `lhs.keys()`

        if hasattr(lhs[k], 'keys'):

            if not hasattr(rhs[k], 'keys'):
                raise TypeError(f'`rhs[{k}]` must be dict-like.')
            else:
                update_lhs_parameters_from_rhs(lhs[k], rhs[k])

        elif hasattr(rhs[k], 'keys'):
            raise TypeError(f'`rhs[{k}]` can not be dict-like.')
        else:
            lhs[k] = rhs[k]
