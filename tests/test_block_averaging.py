import logging

import numpy as np

from hadrian import Hadrian
from hadrian.cli.hadrianalyze import main as hadrianalyze

import pytest


@pytest.fixture
def correlated_data():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(20000, 2, 2))
    decor_factors = np.asarray([[0.0, 0.25], [0.5, 0.998]])
    for i in range(1, data.shape[0]):
        data[i] = decor_factors * data[i - 1] + (1.0 - decor_factors) * data[i]

    return data

@pytest.fixture
def correlated_data_ident_channels():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(20000, 2, 2))
    decor_factors = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    for i in range(1, data.shape[0]):
        data[i] = decor_factors * data[i - 1] + (1.0 - decor_factors) * data[i]

    return data

@pytest.fixture
def oscillating_data():
    rng = np.random.default_rng(5)
    noise = rng.normal(size=(20000, 1))
    decor_factors = 0.5
    for i in range(1, noise.shape[0]):
        noise[i] = decor_factors * noise[i - 1] + (1.0 - decor_factors) * noise[i]

    data = 2 + 2 * np.sin(np.arange(noise.shape[0])).reshape(noise.shape) + noise

    return data

def test_block_averaging_raw():
    rng = np.random.default_rng(5)
    data = rng.uniform(size=(200, 2, 2))

    ht = Hadrian()
    ht.process_traj(data)

    hs = Hadrian()
    for s in data:
        hs.process_sample(s)

    for v_t, v_s in zip(ht.block_sizes, hs.block_sizes):
        assert v_t == v_s, "block_sizes mismatch"
    for v_t, v_s in zip(ht.accum_n_blocks, hs.accum_n_blocks):
        assert v_t == v_s, "accum_n_blocks mismatch"
    for v_t, v_s in zip(ht.accum_block_mean_val_sum, hs.accum_block_mean_val_sum):
        assert np.allclose(v_t, v_s), f"accum_block_mean_val_sums mismatch {v_t - v_s}"
    for v_t, v_s in zip(ht.accum_block_mean_sq_sum, hs.accum_block_mean_sq_sum):
        assert np.allclose(v_t, v_s), f"accum_block_mean_sq_sums mismatch {v_t - v_s}"

def test_block_averaging_proc(correlated_data, tmp_path):
    data = correlated_data

    ht = Hadrian()
    ht.process_traj(data)

    means, std_errs, decor_t = ht.means_and_std_errs(plot_file=tmp_path / "plot_traj.pdf")
    assert means.shape == data[0].shape

    assert np.allclose(means, 0.0, atol=0.02)
    assert np.allclose(std_errs, [[0.007, 0.007], [0.007, -0.01]], atol=0.002)
    # same sign
    assert np.all(std_errs * decor_t > 0)

    hs = Hadrian()
    for d in data:
        hs.process_sample(d)

    s_means, s_std_errs, s_decor_t = hs.means_and_std_errs(plot_file=tmp_path / "plot_sample.pdf")
    assert np.all(means == s_means)
    assert np.all(std_errs == s_std_errs)
    assert np.all(decor_t == s_decor_t)

def test_block_averaging_scalar(correlated_data, tmp_path):
    data = correlated_data[:, 0, 0]
    assert len(data.shape) == 1

    ht = Hadrian()
    ht.process_traj(data)

    means, std_errs, decor_t = ht.means_and_std_errs(plot_file=tmp_path / "plot_traj.pdf")
    assert means.shape == data[0].shape

    assert np.allclose(means, 0.0, atol=0.02)
    assert np.allclose(std_errs, 0.007, atol=0.002)
    # same sign
    assert np.all(std_errs * decor_t > 0)

    hs = Hadrian()
    for d in data:
        hs.process_sample(d)

    s_means, s_std_errs, s_decor_t = hs.means_and_std_errs(plot_file=tmp_path / "plot_sample.pdf")
    assert np.max(np.abs(means - s_means)) < 1e-14
    assert np.max(np.abs(std_errs - s_std_errs)) < 1e-14
    assert np.max(np.abs(decor_t - s_decor_t)) < 1e-14

def test_block_averaging_ident_channels(correlated_data_ident_channels, tmp_path):
    data = correlated_data_ident_channels

    ht = Hadrian(identical_channels=True)
    ht.process_traj(data)

    means, std_errs, decor_t = ht.means_and_std_errs(plot_file=tmp_path / "plot_traj.pdf")

    assert np.allclose(means, 0.0, atol=0.02)
    assert np.allclose(std_errs, 0.007 / np.sqrt(4), atol=0.002)
    # same sign
    assert np.all(std_errs * decor_t > 0)

    hs = Hadrian(identical_channels=True)
    for d in data:
        hs.process_sample(d)

    s_means, s_std_errs, s_decor_t = hs.means_and_std_errs(plot_file=tmp_path / "plot_sample.pdf")
    assert np.all(means == s_means)
    assert np.all(std_errs == s_std_errs)
    assert np.all(decor_t == s_decor_t)

def test_cli(correlated_data, tmp_path, capsys):
    np.savetxt(tmp_path / "data", correlated_data.reshape((correlated_data.shape[0], -1)))

    args = [str(tmp_path / "data")]

    hadrianalyze(args)

    captured = capsys.readouterr().out
    d = np.asarray([np.fromstring(line, sep=' ') for line in captured.splitlines()[2:]])

    assert np.allclose(d[:, 1], 0.0, atol=0.02)
    assert np.allclose(d[:, 2], [0.007, 0.007, 0.007, -0.01], atol=0.002)
    assert np.all(d[:, 2] * d[:, 3] > 0)

def test_oscillating(oscillating_data, tmp_path):
    data = oscillating_data

    ht = Hadrian()
    ht.process_traj(data)

    means, std_errs, decor_t = ht.means_and_std_errs(plot_file=tmp_path / "plot_traj.pdf")

    assert np.allclose(means, 2, atol=0.02)
    # unconverged, negative, due to large apparent std err for smallest block sizes
    assert np.allclose(std_errs, -0.008, atol=0.002)

def test_wrong_shape(correlated_data, tmp_path):
    data = correlated_data
    data = data.T.reshape((-1, data.shape[0]))

    h = Hadrian()
    with pytest.raises(ValueError):
        h.process_traj(data)
        channel_means, channel_std_errs, channel_decorrelation_times = h.means_and_std_errs(plot_file="T_decorrelation.pdf")


    h = Hadrian()
    with pytest.raises(ValueError):
        for d in data:
            h.process_sample(d)
        channel_means, channel_std_errs, channel_decorrelation_times = h.means_and_std_errs(plot_file="T_decorrelation.pdf")

def test_marginal_n_blocks(correlated_data, tmp_path, caplog):
    data = correlated_data
    data = data[0:100]

    h = Hadrian()
    with caplog.at_level(logging.WARNING):
        h.process_traj(data)
        channel_means, channel_std_errs, channel_decorrelation_times = h.means_and_std_errs(plot_file="T_decorrelation.pdf")
        assert 'marginal' in caplog.text

    h = Hadrian()
    with caplog.at_level(logging.WARNING):
        for d in data:
            h.process_sample(d)
        channel_means, channel_std_errs, channel_decorrelation_times = h.means_and_std_errs(plot_file="T_decorrelation.pdf")
        assert 'marginal' in caplog.text
