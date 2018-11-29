
def update_parameters(lhs, rhs):
    '''Update values of keys in dict-like `lhs` with the values of those keys
    in dict-like `rhs`. `lhs` and `rhs` must both have `keys` attribute.'''

    assert hasattr(lhs, 'keys')
    assert hasattr(rhs, 'keys')

    for k in rhs:

        if k not in lhs:
            # Use traceback to query `lhs`
            raise KeyError(k)

        if hasattr(lhs[k], 'keys'):

            if not hasattr(rhs[k], 'keys'):
                raise TypeError(f'`rhs[{k}]` must be dict-like.')
            else:
                update_parameters(lhs[k], rhs[k])

        elif hasattr(rhs[k], 'keys'):
            raise TypeError(f'`rhs[{k}]` can not be dict-like.')
        else:
            lhs[k] = rhs[k]
