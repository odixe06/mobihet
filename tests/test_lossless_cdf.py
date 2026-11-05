import numpy as np
import pytest
from methods.mobihet_lossless import encode, decode, HEADER


def test_cdf_lossless_and_corruption():
    rng = np.random.default_rng(42)
    codes = np.sort(rng.integers(0, 65536, (16, 77, 256), dtype=np.uint16), axis=-1)
    codes[..., -1] = 65535
    packet = encode(codes)
    np.testing.assert_array_equal(decode(packet), codes)
    corrupt = bytearray(packet)
    corrupt[HEADER.size + 5] ^= 1
    with pytest.raises(AssertionError):
        decode(corrupt)
    with pytest.raises(AssertionError):
        decode(packet[:-1])
