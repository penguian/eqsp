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
    if n_caps != len(c_regions):
        raise ValueError(
            "LOOKUP_S2_REGION: Mismatch between length of s_cap and c_regions"
        )
    # The last element of c_regions should be the total number of regions (N)
    n_regions = s_regions.shape[2]
    if c_regions[n_caps - 1] != n_regions:
        raise ValueError(
            "LOOKUP_S2_REGION: Mismatch between c_regions[-1] and length of s_regions"
        )

    # cap_bound_lats: the end colatitude of each cap band
    cap_bound_lats = np.round(s_cap, 12)
    c_start_indices = np.concatenate(([0], c_regions[:-1]))

    n_points = s_point.shape[1]
    r_idx = np.zeros(n_points, dtype=int)
    if n_points == 0:
        return r_idx

    active_c_idxs = np.searchsorted(
        cap_bound_lats, np.round(s_point[1, :], 12), side="left"
    )
    n_caps_detected = len(cap_bound_lats)
    active_c_idxs = np.clip(active_c_idxs, 0, n_caps_detected - 1)

    for c_idx in np.unique(active_c_idxs):
        mask = active_c_idxs == c_idx
        pts_long = s_point[0, mask]
        pts_idx = np.where(mask)[0]

        min_r_idx = int(c_start_indices[c_idx]) + 1

        # Calculate number of sectors in this cap band
        if c_idx < n_caps_detected - 1:
            n_longs = int(c_start_indices[c_idx + 1]) - (min_r_idx - 1)
        else:
            n_longs = n_regions - (min_r_idx - 1)

        if n_longs > 1:
            # --- Point Translation in Monotonic Domain ---
            start_off = min_r_idx - 1
            s_longs = s_regions[0, :, start_off : start_off + n_longs]
            ends = s_longs[1, :]
            starts = s_longs[0, :]

            # Map everything into the range [0, 2*pi)
            phi0 = starts[0]
            two_pi = 2 * np.pi

            # Translate point longitudes to [0, 2*pi)
            pts_long_translated = (pts_long - phi0) % two_pi

            # Translate boundaries (ends) to [0, 2*pi)
            ends_translated = (ends - phi0) % two_pi

            # The ends are monotonically increasing in [0, 2*pi) EXCEPT for the last one
            # which wraps around to exactly 0 (since it's phi0 + 2*pi).
            # We set it to 2*pi to maintain monotonicity for searchsorted.
            if ends_translated[-1] <= 1e-15:
                ends_translated[-1] = two_pi

            # Direct searchsorted on the monotonic translated table
            l_idx = np.atleast_1d(
                np.searchsorted(ends_translated, pts_long_translated, side="left")
            )

            # Handle points exactly on or exceeding the translated 2-pi boundary
            l_idx[l_idx >= n_longs] = 0

            r_idx[pts_idx] = min_r_idx + l_idx
        else:
            # Pole cap logic (single region)
            r_idx[pts_idx] = min_r_idx

    return r_idx
