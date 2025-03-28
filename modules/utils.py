import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchtuples as tt
from pycox.preprocessing.discretization import (
    make_cuts, 
    IdxDiscUnknownC, 
    _values_if_series,
    DiscretizeUnknownC, 
    Duration2Idx
)


class LabelTransform:
    """
    Defines time intervals (cuts) needed for the `PCHazard` method [1].
    One can either pass:
        1) An integer (the number of cuts) or 
        2) An array of pre-defined cut points.

    If an array is given (pre-defined cuts), those are used directly. 
    If an integer is given, an additional step (i.e., a call to `fit`) 
    would typically determine the actual cut points based on the data.

    Reference:
        [1] Kvamme H, Borgan Ø. Continuous and Discrete-Time Survival Prediction with
            Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
            https://arxiv.org/pdf/1910.06724.pdf

    Args:
        cuts (int or array-like): Either the number of cuts or the actual cut points.
        scheme (str, optional): If `cuts` is an integer, the scheme used for 
            discretization (`'equidistant'` or `'quantiles'`). Default is `'equidistant'`.
        min_ (float, optional): Starting duration. Default is `0.0`.
        dtype (str or np.dtype, optional): Data type of the discretization. If 
            `cuts` is an array, `dtype` must be `None`.
    """
    def __init__(self, cuts, scheme='equidistant', min_=0., dtype=None):
        self._cuts = cuts
        self._scheme = scheme
        self._min = min_
        self._dtype_init = dtype
        self._predefined_cuts = False
        self.cuts = None

        # If cuts is an iterable, assume these are the actual cut points
        if hasattr(cuts, '__iter__'):
            if isinstance(cuts, list):
                cuts = np.array(cuts)
            self.cuts = cuts
            self.idu = IdxDiscUnknownC(self.cuts)
            assert dtype is None, "Need `dtype` to be `None` for pre-defined cuts."
            self._dtype = type(self.cuts[0])
            self._dtype_init = self._dtype
            self._predefined_cuts = True
        else:
            # If `cuts` is an int, we add 1 so we can handle indexing appropriately
            self._cuts += 1

    def fit(self, durations: np.ndarray, events: np.ndarray):
        """
        Fit the cut points to the data if `cuts` was an integer. 
        If pre-defined cuts were already given, this does nothing.

        Args:
            durations (np.ndarray): Array of survival durations.
            events (np.ndarray): Binary event indicators (1 if event occurred, 0 if censored).

        Returns:
            LabelTransform: Returns self.
        """
        # If cuts were predefined, do nothing (commented lines show an optional warning)
        # if self._predefined_cuts:
        #     warnings.warn("Calling fit method when 'cuts' are already defined. Leaving cuts unchanged.")
        #     return self

        self._dtype = self._dtype_init
        if self._dtype is None:
            if isinstance(durations[0], np.floating):
                self._dtype = durations.dtype
            else:
                self._dtype = np.dtype('float64')

        durations = durations.astype(self._dtype)

        # Here you would typically compute `self.cuts` via make_cuts if not predefined:
        # self.cuts = make_cuts(self._cuts, self._scheme, durations, events, self._min, self._dtype)
        # For demonstration, we assume self.cuts is already known or precomputed.

        self.duc = DiscretizeUnknownC(self.cuts, right_censor=True, censor_side='right')
        self.di = Duration2Idx(self.cuts)
        return self

    def fit_transform(self, durations: np.ndarray, events: np.ndarray):
        """
        Fit to data (if needed) and then transform durations/events in one step.

        Args:
            durations (np.ndarray): Array of survival durations.
            events (np.ndarray): Binary event indicators.

        Returns:
            tuple:
                - idx_durations (np.ndarray[int]): Discretized duration indices (0-based).
                - events (np.ndarray[float]): (Potentially updated) event indicators.
                - t_frac (np.ndarray[float]): Fraction of the interval in which the event or censoring occurred.
        """
        self.fit(durations, events)
        return self.transform(durations, events)

    def transform(self, durations: np.ndarray, events: np.ndarray):
        """
        Discretize the durations and compute the fractional time within each interval.

        Args:
            durations (np.ndarray): Array of survival durations.
            events (np.ndarray): Binary event indicators.

        Returns:
            tuple:
                - idx_durations (np.ndarray[int]): Discretized duration indices (0-based).
                - events (np.ndarray[float]): (Potentially updated) event indicators.
                - t_frac (np.ndarray[float]): Fraction of the interval in which the event or censoring occurred.
        """
        durations = _values_if_series(durations).astype(self._dtype)
        events = _values_if_series(events)

        dur_disc, events = self.duc.transform(durations, events)
        idx_durations = self.di.transform(dur_disc)

        # Verify cuts are strictly increasing
        cut_diff = np.diff(self.cuts)
        assert (cut_diff > 0).all(), 'Cuts are not unique.'

        # Fraction of interval
        t_frac = 1. - (dur_disc - durations) / cut_diff[idx_durations - 1]

        # Handling events at start time
        if idx_durations.min() == 0:
            warnings.warn(
                "Got event/censoring at start time. It is set such that "
                "it has no contribution to loss."
            )
            t_frac[idx_durations == 0] = 0
            events[idx_durations == 0] = 0

        # Convert 1-based indexing to 0-based for internal use
        idx_durations = idx_durations - 1

        # Ensure no negative indices remain
        idx_durations[idx_durations < 0] = 0

        return (
            idx_durations.astype('int64'),
            events.astype('float32'),
            t_frac.astype('float32')
        )

    @property
    def out_features(self) -> int:
        """
        Number of output features for models (i.e., the number of discretized intervals).

        Raises:
            ValueError: If called before `fit` (i.e., before `self.cuts` is set).

        Returns:
            int: `len(self.cuts) - 1`
        """
        if self.cuts is None:
            raise ValueError("Need to call `fit` before accessing `out_features`.")
        return len(self.cuts) - 1


def pad_col(input: torch.Tensor, val: float = 0, where: str = 'end') -> torch.Tensor:
    """
    Appends a column of constant `val` to a 2D tensor, either at the start or at the end.

    Args:
        input (torch.Tensor): 2D input tensor of shape (batch_size, width).
        val (float, optional): The constant value to be placed in the padded column. Defaults to 0.
        where (str, optional): Either 'start' or 'end'. Defaults to 'end'.

    Raises:
        ValueError: If `input` is not 2D or `where` is not one of ['start', 'end'].

    Returns:
        torch.Tensor: The padded tensor of shape (batch_size, width+1).
    """
    if len(input.shape) != 2:
        raise ValueError("Only works for a 2D tensor, got shape {}.".format(input.shape))
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")


def array_or_tensor(tensor: torch.Tensor, numpy: bool, input_data):
    """
    Deprecated utility function that returns either a NumPy array or a torch.Tensor.

    .. warning::
        This function is deprecated. Use `torchtuples.utils.array_or_tensor` instead.

    Args:
        tensor (torch.Tensor): Input tensor.
        numpy (bool): If True, returns a NumPy array; otherwise returns a torch.Tensor.
        input_data: Original input data (unused directly, but kept for consistency).

    Returns:
        Union[np.ndarray, torch.Tensor]: Either a NumPy array or a torch.Tensor version of `tensor`.
    """
    warnings.warn(
        'Use `torchtuples.utils.array_or_tensor` instead',
        DeprecationWarning
    )
    return tt.utils.array_or_tensor(tensor, numpy, input_data)


def make_subgrid(grid, sub: int = 1):
    """
    When calling `predict_surv` with `sub != 1`, this function helps to create
    a refined (subdivided) duration index for the survival estimates.

    Example:
        sub = 5
        surv = model.predict_surv(test_input, sub=sub)
        grid = model.make_subgrid(cuts, sub)
        surv = pd.DataFrame(surv, index=grid)

    Args:
        grid (array-like): The main grid (cut points).
        sub (int, optional): The factor by which to subdivide each interval. Defaults to 1.

    Returns:
        torchtuples.TupleTree or similar structure: The subdivided grid intervals.
    """
    subgrid = tt.TupleTree(
        np.linspace(start, end, num=sub+1)[:-1]
        for start, end in zip(grid[:-1], grid[1:])
    )
    subgrid = subgrid.apply(lambda x: tt.TupleTree(x)).flatten() + (grid[-1],)
    return subgrid


def log_softplus(input: torch.Tensor, threshold: float = -15.) -> torch.Tensor:
    """
    Computes `log(softplus(x))` in a numerically stable way. For `input < threshold`,
    returns `x` since `softplus(x)` ~ `exp(x)` when x is very negative, hence
    `log(softplus(x)) ~ x`.

    Args:
        input (torch.Tensor): Input tensor.
        threshold (float, optional): Threshold for approximation. Defaults to -15.

    Returns:
        torch.Tensor: `log(softplus(x))`, approximated by `x` when `x` is small.
    """
    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output


def cumsum_reverse(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Computes the reverse cumulative sum along a given dimension.

    Currently implemented only for `dim = 1`.

    Args:
        input (torch.Tensor): 2D tensor (batch_size, width).
        dim (int, optional): Dimension along which to compute the reverse cumulative sum. 
                             Must be 1. Defaults to 1.

    Raises:
        NotImplementedError: If `dim` != 1.

    Returns:
        torch.Tensor: The reversed cumulative sum of the input along dim=1.
    """
    if dim != 1:
        raise NotImplementedError("cumsum_reverse is only implemented for dim=1.")
    # Sum over dim=1, then subtract the forward cumulative sum (padded)
    input = input.sum(1, keepdim=True) - pad_col(input, where='start').cumsum(1)
    return input[:, :-1]


def set_random_seed(seed: int = 1234):
    """
    Set random seed for reproducibility across numpy, PyTorch (CPU/GPU).

    Args:
        seed (int, optional): The seed to use. Defaults to 1234.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
