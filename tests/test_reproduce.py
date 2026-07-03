"""Stimulus-noise reproduction — MATLAB-anchored regression tests.

Reference values were generated in MATLAB R2023b with the exact recipe the stimulus
scripts use (mt19937ar + norminv(rand), column-major), so these pin the Python
reproduction to MATLAB bit-for-bit.
"""
import numpy as np
from neitz.stimulus import reproduce_noise, gamma_adjust, expand_to_frames

# MATLAB: rand(RandStream('mt19937ar','Seed',2), 1, 6) — the uniform stream that
# underlies every reproduction. numpy's legacy RandomState must match it exactly.
MATLAB_RAND_SEED2 = np.array([
    0.435994902142, 0.025926231828, 0.549662477879,
    0.435322392618, 0.420367802087, 0.330334821004])

# MATLAB: nv(:) for nv = 0.5 + 0.3*norminv(rand(seed=7, 3,4,2)), column-major flatten,
# UNCLIPPED (deliberately includes 1.104 and -0.089 so the clip is exercised).
MATLAB_SEED7_3x4x2_COLMAJOR = np.array([
    0.070894987161453205, 0.73157569381852039,  0.4534989137826908,  0.67794998825370634,
    1.1041673075947638,   0.52899352256063825,  0.50084257687494016, 0.061794892619500053,
    0.31473770612450036,  0.49991164197285337,  0.6396640076415353,  0.75651583472917094,
    0.40909701005712579,  0.047972594227208787, 0.332356912804879,   0.90147685076997486,
    0.26158106687111859,  0.46391091633302822,  0.94545002140553058, -0.088507339121486783,
    0.5764304493641651,   0.99383316732501359,  0.27864507525840104, 0.53655411101965578])


def test_uniform_stream_matches_matlab():
    # Foundation of cross-language reproducibility: numpy RandomState == MATLAB rand.
    assert np.allclose(np.random.RandomState(2).rand(6), MATLAB_RAND_SEED2, atol=1e-11)


def test_reproduce_matches_matlab_reference():
    v = reproduce_noise(7, 3, 4, 2, mu=0.5, sigma=0.3)
    assert v.shape == (3, 4, 2)
    expected = np.clip(MATLAB_SEED7_3x4x2_COLMAJOR, 0.0, 1.0)   # reproduce_noise clips
    assert np.allclose(v.flatten(order="F"), expected, atol=1e-12)


def test_clip_applied():
    # raw reference had 1.104 and -0.089 -> tensor must be clipped to [0, 1]
    v = reproduce_noise(7, 3, 4, 2)
    assert v.min() >= 0.0 and v.max() <= 1.0
    assert np.isclose(v.max(), 1.0) and np.isclose(v.min(), 0.0)


def test_shape_fullfield_and_checkerboard():
    assert reproduce_noise(2, 1, 1, 6).shape == (1, 1, 6)       # full-field
    assert reproduce_noise(2, 32, 40, 5).shape == (32, 40, 5)   # 40x32 checkerboard


def test_determinism():
    assert np.array_equal(reproduce_noise(123, 8, 8, 4), reproduce_noise(123, 8, 8, 4))


def test_gamma_adjust_known_codes():
    codes = np.round(255 * gamma_adjust(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))).astype(int)
    assert list(codes) == [0, 136, 186, 224, 255]
    assert gamma_adjust(0.0) == 0.0 and np.isclose(gamma_adjust(1.0), 1.0)   # fixed points


def test_expand_to_frames_holds_updates():
    upd = reproduce_noise(2, 2, 3, 4)                       # (2, 3, 4)
    frames = expand_to_frames(upd, update_every_n_frames=8, stim_frames=30)
    assert frames.shape == (2, 3, 30)
    assert np.array_equal(frames[..., 0], upd[..., 0])     # held...
    assert np.array_equal(frames[..., 7], upd[..., 0])
    assert np.array_equal(frames[..., 8], upd[..., 1])     # ...then next update
