"""
Partition options for EQ partition.
"""

def partition_options(pdefault, *varargin):
    """
    popt = partition_options(pdefault, *varargin)

    Collects partition options, specified as name, value pairs, and places these
    into the dictionary popt. The dictionary pdefault is used to define default
    option values.

    Parameters
    ----------
    pdefault : dict
        Default options.
    *varargin
        Options.

    Returns
    -------
    popt : dict
        Combined options.

    Notes
    -----
    The structures pdefault and popt may contain the following fields:
    extra_offset : bool

    The following partition options are available:

    'offset' : Control extra rotation offsets for S^2 and S^3 regions.
        'extra'  : Use extra rotation offsets, sets extra_offset to True.
        'normal' : Do not use extra offsets, sets extra_offset to False.

    Shortcuts:
    partition_options(pdefault, 'extra') -> extra_offset = True
    partition_options(pdefault, 'normal') -> extra_offset = False
    partition_options(pdefault, True) -> extra_offset = True
    partition_options(pdefault, False) -> extra_offset = False

    Examples
    --------
    >>> pdefault = {'extra_offset': False}
    >>> popt = partition_options(pdefault, 'offset', 'extra')
    >>> popt['extra_offset']
    True
    >>> popt = partition_options(pdefault, False)
    >>> popt['extra_offset']
    False
    """
    popt = pdefault.copy()
    args = list(varargin)
    nargs = len(args)

    if nargs == 1:
        # Short circuit: single argument is value of extra_offset
        value = args[0]
        if value is True:
            popt['extra_offset'] = True
        elif value is False:
            popt['extra_offset'] = False
        elif value == 'extra':
            popt['extra_offset'] = True
        elif value == 'normal':
            popt['extra_offset'] = False
        else:
            raise ValueError(f"Invalid option value: {value}")
        return popt

    # Process pairs
    i = 0
    while i < nargs:
        arg = args[i]
        if isinstance(arg, str):
            if arg == 'offset':
                if i + 1 < nargs:
                    value = args[i+1]
                    if value == 'extra':
                        popt['extra_offset'] = True
                    elif value == 'normal':
                        popt['extra_offset'] = False
                    else:
                         raise ValueError(f"Invalid option value for offset: {value}")
                    i += 2
                else:
                    raise ValueError("Option 'offset' requires a value")
            else:
                 # Ignore unknown text options or raise error?
                 # Matlab code seems to check structure of args but iterates loosely.
                 # For now, let's just skip unknown string keys + value if they look like pairs?
                 # Actually, Matlab code splits args into keys and values assuming pairs.
                 # Let's assume pairs if string.
                 i += 2
        else:
             # Skip non-string arguments?
             i += 1
    
    return popt
