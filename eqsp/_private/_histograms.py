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
    >>> from eqsp._private._histograms import lookup_s2_region
    >>> import eqsp
    >>> points_s = eqsp.eq_point_set_polar(2, 8)
    >>> N = 8
    >>> s_regions = eqsp.eq_regions(2, N)
    >>> s_cap, n_regions = eqsp.eq_caps(2, N)
    >>> c_regions = np.cumsum(n_regions)
    >>> r_idx = lookup_s2_region(points_s, s_regions, s_cap, c_regions)
    >>> print(r_idx)
    [1 2 3 4 5 6 7 8]
    >>> N = 5
    >>> s_regions = eqsp.eq_regions(2, N)
    >>> s_cap, n_regions = eqsp.eq_caps(2, N)
    >>> c_regions = np.cumsum(n_regions)
    >>> r_idx = lookup_s2_region(points_s, s_regions, s_cap, c_regions)
    >>> print(r_idx)
    [1 2 2 3 3 4 4 5]
    """
    import numpy as np

    n_caps = len(s_cap)
    assert n_caps == len(c_regions), "LOOKUP_S2_REGION: Mismatch between length of s_cap and c_regions"
    
    n_regions = s_regions.shape[2]
    assert c_regions[n_caps - 1] == n_regions, "LOOKUP_S2_REGION: Mismatch between c_regions[-1] and length of s_regions"
    n_points = s_point.shape[1]
    r_idx = np.zeros(n_points, dtype=int)
    if n_points == 0:
        return r_idx

    c_idx_all = np.atleast_1d(lookup_table(s_cap, s_point[1, :]))

    r_idx[c_idx_all == 0] = 1
    r_idx[c_idx_all >= n_caps - 1] = n_regions

    active_mask = (c_idx_all > 0) & (c_idx_all < n_caps - 1)
    if np.any(active_mask):
        active_c_idxs = c_idx_all[active_mask]
        active_longs = s_point[0, active_mask]
        orig_indices = np.where(active_mask)[0]

        for c_idx in np.unique(active_c_idxs):
            collar_mask = (active_c_idxs == c_idx)
            pts_long = active_longs[collar_mask]
            pts_idx = orig_indices[collar_mask]

            min_r_idx = int(c_regions[c_idx - 1]) + 1
            max_r_idx = int(c_regions[c_idx])

            s_longs = s_regions[0, :, min_r_idx - 1 : max_r_idx].copy()
            if s_longs[0, 0] >= 2 * np.pi:
                s_longs[:, 0] -= 2 * np.pi
            n_longs = s_longs.shape[1]

            l_idx = np.atleast_1d(lookup_table(s_longs[1, :], pts_long)) % n_longs
            
            wrap_mask = pts_long < s_longs[0, 0]
            l_idx[wrap_mask] = n_longs - 1

            r_idx[pts_idx] = min_r_idx + l_idx

    return r_idx


def lookup_table(table, y):
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

    # Nondecreasing table.
    maximum = np.max(np.concatenate([table, y])) + 1
    extended_table = np.append(table, maximum)
    idx = np.searchsorted(extended_table, y, side="right")
    idx[idx < 0] = 0
    
    if idx.size == 1:
        return int(idx[0])
    return idx



