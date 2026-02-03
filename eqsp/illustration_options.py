"""
Illustration options for EQ partitions.
"""

def illustration_options(gdefault, *varargin):
    """
    gopt = illustration_options(gdefault, *varargin)

    Collects illustration options, specified as name, value pairs, and places these
    into the dictionary gopt. The dictionary gdefault is used to define default
    option values.

    Parameters
    ----------
    gdefault : dict
        Default options.
    *varargin
        Options.

    Returns
    -------
    gopt : dict
        Combined options.
    """
    gopt = gdefault.copy()
    args = list(varargin)
    nargs = len(args)
    
    # Process pairs
    i = 0
    while i < nargs:
        arg = args[i]
        if not isinstance(arg, str):
            i += 1
            continue
            
        if arg == 'fontsize':
            if i + 1 < nargs:
                gopt['fontsize'] = args[i+1]
                i += 2
            else:
                 raise ValueError("Option 'fontsize' requires a value")
        elif arg == 'title':
            if i + 1 < nargs:
                val = args[i+1]
                if val == 'long':
                    gopt['show_title'] = True
                    gopt['long_title'] = True
                elif val == 'short':
                    gopt['show_title'] = True
                    gopt['long_title'] = False
                elif val in ['none', 'hide']:
                    gopt['show_title'] = False
                    gopt['long_title'] = False
                elif val == 'show':
                    gopt['show_title'] = True
                else:
                    raise ValueError(f"Invalid value for 'title': {val}")
                i += 2
            else:
                raise ValueError("Option 'title' requires a value")
        elif arg == 'proj':
             if i + 1 < nargs:
                val = args[i+1]
                if val == 'stereo':
                    gopt['stereo'] = True
                elif val == 'eqarea':
                    gopt['stereo'] = False
                else:
                    raise ValueError(f"Invalid value for 'proj': {val}")
                i += 2
             else:
                raise ValueError("Option 'proj' requires a value")
        elif arg == 'points':
             if i + 1 < nargs:
                val = args[i+1]
                if val == 'show':
                    gopt['show_points'] = True
                elif val == 'hide':
                    gopt['show_points'] = False
                else:
                    raise ValueError(f"Invalid value for 'points': {val}")
                i += 2
             else:
                raise ValueError("Option 'points' requires a value")
        elif arg == 'sphere':
             if i + 1 < nargs:
                val = args[i+1]
                if val == 'show':
                    gopt['show_sphere'] = True
                elif val == 'hide':
                    gopt['show_sphere'] = False
                else:
                    raise ValueError(f"Invalid value for 'sphere': {val}")
                i += 2
             else:
                raise ValueError("Option 'sphere' requires a value")
        elif arg == 'surf':
             if i + 1 < nargs:
                val = args[i+1]
                if val == 'show':
                    gopt['show_surfaces'] = True
                elif val == 'hide':
                    gopt['show_surfaces'] = False
                else:
                    raise ValueError(f"Invalid value for 'surf': {val}")
                i += 2
             else:
                raise ValueError("Option 'surf' requires a value")
        else:
            # Ignore or increment? Matlab increments by 2 usually if it detects pairs
            i += 2
            
    return gopt
