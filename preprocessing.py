import numpy as np

#%%
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
#%%
def interpolate_nans(t, y, gap_dur=np.inf):
    ''' Replaces nans with interpolated values if the 'island' or nans/or gap
        is shorter than 'gap_dur'

    Args:
        y - 1d numpy array,
        gap_dur - duration of gap in ms

    Returns:
        y - interpolated array
    '''

    # Find index for nans where gaps are longer than 'gap_dur' samples
    d = np.isnan(y)

    # If there are no nans, return
    if not np.any(d):
        return y

    # Find onsets and offsets of gaps
    d = np.diff(np.concatenate((np.array([0]), d*1, np.array([0]))))
    onsets = np.where(d==1)[0]
    offsets = np.where(d==-1)[0]

    # Decrease offsets come too late by -1
    if np.any(offsets >= len(y)):
        idx = np.where(offsets >= len(y))[0][0]
        offsets[idx] = offsets[idx] - 1

    dur = t[offsets] - t[onsets]


    # If the gaps are longer than 'gaps', replace temporarily with other values
    for i, on in enumerate(onsets):
        if dur[i] > gap_dur:
            y[onsets[i]:offsets[i]] = -1000

    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    # put nans back
    y[y == -1000] = np.nan

    return y


#%%
def nearest_odd_integer(x):
    """Returns nearest odd integer of input value.

    Input:
        - x, float or integer
    Output:
        - integer value that is odd
    Example:
        >>> nearest_odd_integer(6.25)
            7
        >>> nearest_odd_integer(5.99)
            5
    """

    return int(2*np.floor(x/2)+1)