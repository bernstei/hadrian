import logging
import numpy as np

from matplotlib.figure import Figure

class Hadrian:
    """calculation of time series data standard errors on means by blocked averaging.  Optimal
    block size calculated using heuristic from R. M. Lee et al., Phys. Rev. B p. 066706, v. 83
    (2011), p. 7, last paragraph of col 1.

    Parameters
    ----------
    min_n_blocks: int, default 20
        minimum number of blocks (related to max block size) to consider in analysis
    identical_channels: bool, default False
        whether channels are identical and should therefore be combined
    block_size_factor: float, default sqrt(2)
        factor between sizes of blocks
    """
    def __init__(self, min_n_blocks=20, identical_channels=False, block_size_factor=np.sqrt(2)):
        self.min_n_blocks = min_n_blocks
        self.identical_channels = identical_channels
        self.block_size_factor = block_size_factor

        # initialize for sample-by-sample processing
        self.n_samples = 0


    def process_traj(self, samples):
        """process an array of samples, computing means in various block sizes

        Parameters
        ----------
        samples: list(ndarray)
            List of samples containing entire trajectory to process, with each sample
            an ndarray of any shape identical to all other samples (note that, therefore,
            if `samples` is a single ndarray, leading index must be number of samples).
        """
        self.n_samples = len(samples)
        self.block_sizes = [1]
        block_size_i = 0
        while True:
            block_size_i += 1
            next_block_size = int(np.round(self.block_size_factor ** block_size_i))
            if next_block_size > self.n_samples // self.min_n_blocks:
                break
            if next_block_size > self.block_sizes[-1]:
                self.block_sizes.append(next_block_size)

        self.sample_shape_orig = samples[0].shape

        self.accum_block_mean_val_sum = []
        self.accum_block_mean_sq_sum = []
        self.accum_n_blocks = []
        for block_size in self.block_sizes:
            n_blocks = self.n_samples // block_size
            trunc_samples = np.asarray(samples[:block_size * n_blocks])
            if len(trunc_samples.shape) == 1:
                trunc_samples = trunc_samples.reshape((trunc_samples.shape[0], 1))

            # slightly fiddly that this particular reshape is needed to get the blocking we want
            new_shape = [n_blocks, block_size] + list(trunc_samples.shape[1:])
            blocked_samples = trunc_samples.reshape(new_shape)

            # mean within each block
            block_means = np.mean(blocked_samples, axis=1)
            # sum over blocks
            self.accum_block_mean_val_sum.append(np.sum(block_means, axis=0))
            self.accum_block_mean_sq_sum.append(np.sum(block_means ** 2, axis=0))
            self.accum_n_blocks.append(n_blocks)

        # data for means_and_std_errs
        self.sample_shape = trunc_samples[0].shape
        self.accum_block_mean_val_sum = np.asarray(self.accum_block_mean_val_sum)
        self.accum_block_mean_sq_sum = np.asarray(self.accum_block_mean_sq_sum)


    def process_sample(self, sample):
        """process a sample, accumulating in various block sizes

        Parameters
        ----------
        sample: ndarray
            sample to process, must be same shape as all previous ones
        """
        self.sample_shape_orig = sample.shape
        if len(sample.shape) == 0:
            sample = sample.reshape((1,))

        if self.n_samples == 0:
            # first sample
            # create empty one-per-block-size data structures for block_size = 1
            self.block_sizes = np.asarray([1])
            self.accum_n_blocks = np.asarray([0])
            self.n_in_block = np.asarray([0])
            # data: first index block size, 2nd ... indices channels in analysis
            # start with one block size per analysis
            self.sample_shape = sample.shape
            data_shape = [1] + list(self.sample_shape)
            self.cur_block_val_sum = np.zeros(data_shape)
            self.accum_block_mean_val_sum = np.zeros(data_shape)
            self.accum_block_mean_sq_sum = np.zeros(data_shape)
            self.next_block_size = 2
            self.next_block_size_exp = 1

        self.n_samples += 1

        # accumulate all existing blocks
        self.cur_block_val_sum += sample
        self.n_in_block += 1

        # process blocks that were filled just now
        filled_blocks = np.where(self.n_in_block == self.block_sizes)[0]
        filled_blocks_means = (self.cur_block_val_sum[filled_blocks].T / self.block_sizes[filled_blocks]).T
        self.accum_block_mean_val_sum[filled_blocks] += filled_blocks_means
        self.accum_block_mean_sq_sum[filled_blocks] += filled_blocks_means ** 2

        self.accum_n_blocks[filled_blocks] += 1
        self.n_in_block[filled_blocks] = 0
        self.cur_block_val_sum[filled_blocks] = 0.0

        # add new blocks if needed
        if self.n_samples == self.next_block_size:
            # new complete block of size self.next_block_size
            self.block_sizes = np.concatenate((self.block_sizes, [self.next_block_size]))
            self.accum_n_blocks = np.concatenate((self.accum_n_blocks, [1]))
            self.n_in_block = np.concatenate((self.n_in_block, [0]))

            # value to store as "accum" block, from accumulated mean of block index 0 that has size 1
            assert self.block_sizes[0] == 1
            new_block_val_mean = self.accum_block_mean_val_sum[0] / self.next_block_size

            self.accum_block_mean_val_sum = np.concatenate((self.accum_block_mean_val_sum, [new_block_val_mean]))
            self.accum_block_mean_sq_sum = np.concatenate((self.accum_block_mean_sq_sum, [new_block_val_mean ** 2]))

            # create zeros of correct shape for accumulating sum for new block size
            self.cur_block_val_sum = np.concatenate((self.cur_block_val_sum, [np.zeros(self.sample_shape)]))

            while True:
                self.next_block_size = int(np.round(self.block_size_factor ** self.next_block_size_exp))
                self.next_block_size_exp += 1
                if self.next_block_size > self.block_sizes[-1]:
                    break


    def means_and_std_errs(self, plot_file=None, convergence_tolerance=3.0):
        """Calculate means and std errs on means from processed correlated sampling data
        using block averaging. 

        Parameters
        ----------
        plot_file: str / Path, default None
            path of plot used to gauge whether convergence has been achieved
        convergence_tolerance: float, default 2.0
            factor applied to error bars to determine if trajectory
            is long enough for error bar to be converged; larger is
            more tolerant

        Returns
        -------
        means: ndarray with same shape as samples containing mean for each channel.
        std_errs: ndarray with same shape as samples containing estimate of standard
                  errors on each mean. Negative values indicate that convergence has
                  not been reached.
        decorrelation_times: ndarray with same shape as samples containing effective
                             decorrelation time for each channel. Negative values indicate
                             that convergence has not been reached.

        """
        if len(self.block_sizes) < 2:
            raise ValueError(f"Trajectory length {self.n_samples} resulted in too few block sizes "
                             f"{self.block_sizes}, number of channels is {self.sample_shape}")
        elif len(self.block_sizes) < 10:
            logging.warning(f"Trajectory length {self.n_samples} resulted in a marginally low number "
                            f"of block sizes {self.block_sizes}, number of channels is {self.sample_shape}")

        self.accum_n_blocks = np.asarray(self.accum_n_blocks)
        self.block_sizes = np.asarray(self.block_sizes)

        have_enough_blocks = np.where(self.accum_n_blocks >= self.min_n_blocks)[0]

        self.block_sizes = self.block_sizes[have_enough_blocks]
        self.accum_n_blocks = self.accum_n_blocks[have_enough_blocks]
        self.accum_block_mean_val_sum = self.accum_block_mean_val_sum[have_enough_blocks]
        self.accum_block_mean_sq_sum = self.accum_block_mean_sq_sum[have_enough_blocks]

        n_block_sizes = len(self.block_sizes)

        if self.identical_channels:
            n_channels = np.prod(self.sample_shape)
            self.sample_shape = (1,)
            self.accum_n_blocks *= n_channels
            self.accum_block_mean_val_sum = np.sum(self.accum_block_mean_val_sum.reshape((n_block_sizes, -1)), axis=1).reshape((n_block_sizes, 1))
            self.accum_block_mean_sq_sum = np.sum(self.accum_block_mean_sq_sum.reshape((n_block_sizes, -1)), axis=1).reshape((n_block_sizes, 1))

        d_means = (self.accum_block_mean_val_sum.T / self.accum_n_blocks).T
        d_vars = (self.accum_block_mean_sq_sum.T / self.accum_n_blocks).T - d_means ** 2
        d_mean_std_errs = np.sqrt((d_vars.T / (self.accum_n_blocks - 1)).T)
        d_mean_std_err_errs = (d_mean_std_errs.T / np.sqrt(self.accum_n_blocks - 1)).T

        # Determine optimal block size using heuristic from R M Lee et al, Phys. Rev. B 83
        # p. 066706 (2011) (p. 7, last paragraph of col 1)
        opt_block_sizes = np.zeros(self.sample_shape, dtype=int)
        opt_block_size_inds = np.zeros(self.sample_shape, dtype=int)
        # default to largest error, if heuristic isn't satisfied
        std_errs_out = np.max(d_mean_std_errs, axis=0)

        opt_block_size_failed = np.zeros(opt_block_sizes.shape, dtype=bool)
        for block_size_i, (block_size, std_err) in reversed(list(enumerate(zip(self.block_sizes, d_mean_std_errs)))):
            eta_err = std_err / d_mean_std_errs[0]
            block_size_is_valid = block_size ** 3 > 2 * self.n_samples * (eta_err ** 4)

            # reset optimal blocksize if we're still in a contiguous streak of valid block sizes
            reset_val = np.logical_and(block_size_is_valid, ~opt_block_size_failed)
            opt_block_sizes[reset_val] = block_size
            opt_block_size_inds[reset_val] = block_size_i
            std_errs_out[reset_val] = std_err[reset_val]

            opt_block_size_failed = np.logical_or(opt_block_size_failed, ~block_size_is_valid)

        # unconverged if heuristic was never satisfied
        unconverged = opt_block_sizes == 0
        # check that no smaller block sizes has a std. err significantly larger than err for optimal block size
        # "significant" is defined as tolerance factor times sqrt of sum of squares of errors
        std_errs_out_flat = std_errs_out.reshape((-1))
        d_mean_std_errs_flat = d_mean_std_errs.reshape((n_block_sizes, -1))
        d_mean_std_err_errs_flat = d_mean_std_err_errs.reshape((n_block_sizes, -1))
        opt_block_size_inds_flat = opt_block_size_inds.reshape((-1))
        for c_i in range(np.prod(self.sample_shape)):
            unconverged_inds = np.unravel_index(c_i, unconverged.shape)
            if unconverged[unconverged_inds]:
                continue
            max_block_ind = opt_block_size_inds_flat[c_i]
            if np.max(d_mean_std_errs_flat[:max_block_ind, c_i] - std_errs_out_flat[c_i] - 
                      convergence_tolerance * np.sqrt(d_mean_std_err_errs_flat[:max_block_ind, c_i] ** 2 + d_mean_std_err_errs_flat[max_block_ind, c_i] ** 2)) > 0:
                unconverged[unconverged_inds] |= True

        decor_time = (std_errs_out / d_mean_std_errs[0]) ** 2

        if plot_file is not None:
            fig = Figure()
            ax = fig.add_subplot()
            # ax.plot(self.block_sizes, d_mean_std_errs, "-")
            assert n_block_sizes == d_mean_std_errs.shape[0]
            assert n_block_sizes == d_mean_std_err_errs.shape[0]
            d =    d_mean_std_errs.reshape((n_block_sizes, -1))
            derr = d_mean_std_err_errs.reshape((n_block_sizes, -1))
            for c_i, (channel_d, channel_err_scan, channel_unconv, channel_t, channel_err) in enumerate(zip(d.T, derr.T,
                                                                                          unconverged.reshape((-1)),
                                                                                          decor_time.reshape((-1)),
                                                                                          std_errs_out.reshape((-1)))):
                # ax.plot(self.block_sizes, channel_d, '-', label=f"channel {c_i}")
                c_ind = np.unravel_index(c_i, d_mean_std_errs.shape[1:])
                ax.errorbar(self.block_sizes, channel_d, yerr=channel_err_scan, fmt='--' if channel_unconv else '-',
                            color=f"C{c_i}", capsize=5, label=f"channel {c_ind} decor time. {channel_t:.1f} std. err {channel_err:f}")

            ylim = ax.get_ylim()
            for c_i, opt_block_size in enumerate(opt_block_sizes.reshape((-1))):
                ax.plot([opt_block_size * 1.05 ** c_i, opt_block_size * 1.05 ** c_i], ylim, "--", color=f"C{c_i}", label=None)
            ax.set_ylim(*ylim)

            ax.set_xscale('log', base=self.block_size_factor)
            interval = max(1, len(self.block_sizes) // 7)
            ax.set_xticks(self.block_sizes[::interval], [f"{int(s)}" for s in self.block_sizes[::interval]])
            ax.set_xlim(self.block_sizes[0] / self.block_size_factor, self.block_sizes[-1] * self.block_size_factor)
            ax.set_xlabel("block size")
            ax.set_ylabel("apparent std. err. on mean")
            ax.legend()
            fig.savefig(plot_file, bbox_inches='tight')

        conv_factor = np.ones(std_errs_out.shape)
        conv_factor[unconverged] = -1
        if len(self.sample_shape_orig) == 0:
            return d_means[0][0], std_errs_out[0] * conv_factor, decor_time[0] * conv_factor
        else:
            return d_means[0], std_errs_out * conv_factor, decor_time * conv_factor
