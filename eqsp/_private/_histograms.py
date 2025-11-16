def lookup_s2_region(s_point, s_regions, s_cap, c_regions):
    """
    For S^2, given sequences of points, regions, and cap colatitudes, find the
    index of the region containing each point.

    Parameters
    ----------
    s_point : ndarray
        Sequence of points on S^2, as a 2 x n_points array in spherical polar
        coordinates, with longitude 0 <= s[0, p_idx] <= 2 * pi, colatitude
        0 <= s[1, p_idx] <= pi.
    s_regions : ndarray
        Sequence of regions of S^2 as per eq_regions(2, N) where N ==
        s_regions.shape[2].
    s_cap : ndarray
        Sequence of cap colatitudes as per eq_caps(2, N) for the same N.
    c_regions : ndarray
        Sequence of the cumulative number of regions of s_regions within each
        cap of s_cap.

    Returns
    -------
    r_idx : ndarray
        Array of length s_point.shape[1] containing the index of the region of
        s_regions corresponding to each point.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent.

    See Also
    --------
    eq_regions, eq_caps, numpy.cumsum, lookup_table

    Notes
    -----
    For references, see AUTHORS. For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> from eq_histogram.private import lookup_s2_region
    >>> points_s = eq_point_set_polar(2, 8)
    >>> N = 8
    >>> s_regions = eq_regions(2, N)
    >>> s_cap, n_regions = eq_caps(2, N)
    >>> c_regions = np.cumsum(n_regions)
    >>> r_idx = lookup_s2_region(points_s, s_regions, s_cap, c_regions)
    >>> print(r_idx)
    [1 2 3 4 5 6 7 8]
    >>> N = 5
    >>> s_regions = eq_regions(2, N)
    >>> s_cap, n_regions = eq_caps(2, N)
    >>> c_regions = np.cumsum(n_regions)
    >>> r_idx = lookup_s2_region(points_s, s_regions, s_cap, c_regions)
    >>> print(r_idx)
    [1 2 2 3 3 4 4 5]
    """
    import numpy as np

    n_caps = len(s_cap)
    if n_caps != len(c_regions):
        msg = (
            "LOOKUP_S2_REGION: Mismatch between length of s_cap (=={}) and "
            "length of c_regions (=={})"
        )
        print(msg.format(n_caps, len(c_regions)))
        raise ValueError(msg.format(n_caps, len(c_regions)))
    n_regions = s_regions.shape[2]
    if c_regions[n_caps - 1] != n_regions:
        msg = (
            "LOOKUP_S2_REGION: Mismatch between c_regions[-1] (=={}) and "
            "length of s_regions (=={})"
        )
        print(msg.format(c_regions[n_caps - 1], n_regions))
        raise ValueError(msg.format(c_regions[n_caps - 1], n_regions))
    n_points = s_point.shape[1]
    r_idx = np.zeros(n_points, dtype=int)
    for p_idx in range(n_points):
        # Lookup by colatitude.
        c_idx = lookup_table(s_cap, s_point[1, p_idx])
        if c_idx > 0 and c_idx < n_caps - 1:
            min_r_idx = c_regions[c_idx] + 1
            max_r_idx = c_regions[c_idx + 1]
            s_longs = np.squeeze(s_regions[0:2, :, min_r_idx:max_r_idx + 1])
            if s_longs[0, 0] >= 2 * np.pi:
                s_longs[0, 0] -= 2 * np.pi
            n_longs = s_longs.shape[1]
            # Lookup by longitude.
            l_idx = lookup_table(s_longs[1, :], s_point[0, p_idx]) % n_longs
            if s_point[0, p_idx] < s_longs[0, 0]:
                l_idx = n_longs - 1
            r_idx[p_idx] = min_r_idx + l_idx
        elif c_idx == 0:
            r_idx[p_idx] = 1
        elif c_idx >= n_caps - 1:
            r_idx[p_idx] = n_regions
        else:
            r_idx[p_idx] = 0
    return r_idx


def lookup_table(table, y, opt=None):
    """
    Lookup values in a sorted table. Usually used as a prelude to interpolation.

    If table is strictly increasing and idx = lookup_table(table, y), then
    table[idx[i]] <= y[i] < table[idx[i]+1] for all y[i] within the interval
    with minimum table[0] and maximum table[-1]. If y[i] < table[0] then
    idx[i] is 0. If y[i] >= table[-1] then idx[i] is len(table) - 1.

    If the table is strictly decreasing, then the tests are reversed.
    (NOT YET IMPLEMENTED). There are no guarantees for tables which are
    non-monotonic or are not strictly monotonic.

    Parameters
    ----------
    table : array_like
        Sequence of real values, assumed to be strictly increasing or decreasing.
    y : array_like or float
        Sequence of real values to be looked up in table.
    opt : any, optional
        Options to be passed to Octave lookup, if we are in Octave, otherwise
        ignored.

    Returns
    -------
    idx : int or ndarray
        Indices in table corresponding to each y.

    Raises
    ------
    NotImplementedError
        If table is strictly decreasing.

    See Also
    --------
    numpy.searchsorted

    Notes
    -----
    For references, see AUTHORS. For revision history, see CHANGELOG.

    Examples
    --------
    >>> table = [-100.0, -70, 2.5, 75, 125.7]
    >>> y = [-1, 3, 1000, -197]
    >>> lookup_table(table, y)
    array([2, 3, 5, 0])
    """
    import numpy as np

    table = np.asarray(table)
    y = np.atleast_1d(y)
    if is_octave():
        from octave_functions import lookup  # Assume importable
        if opt is not None:
            idx = lookup(table, y, opt)
        else:
            idx = lookup(table, y)
    else:
        if table[-1] >= table[0]:
            # Nondecreasing table.
            maximum = np.max(np.concatenate([table, y])) + 1
            extended_table = np.append(table, maximum)
            idx = np.array(
                [np.searchsorted(extended_table, val, side="left") - 1 for val in y]
            )
        else:
            print("LOOKUP_TABLE: Decreasing case is not yet implemented")
            raise NotImplementedError(
                "Strictly decreasing tables are not implemented."
            )
    if idx.size == 1:
        return idx[0]
    return idx


def is_octave():
    """
    Checks if running in Octave.

    Returns
    -------
    bool
        True if in Octave, False otherwise.

    See Also
    --------
    octave_functions

    Examples
    --------
    >>> is_octave()
    False
    """
    try:
        import octave_functions  # Assume importable
        return True
    except ImportError:
        return False
