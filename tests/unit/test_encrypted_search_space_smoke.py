import argparse

import pytest

from dev.smoke_test_encrypted_search_space import _sha256_value


SHA256 = "9a232adcc36a6451a4a30f3fe1dbfb29c4476b7b324bcec87bc0f5cc30bbf70d"


@pytest.mark.parametrize("value", [SHA256, f"sha256:{SHA256}", SHA256.upper()])
def test_sha256_value_accepts_release_digest_forms(value):
    assert _sha256_value(value) == SHA256


@pytest.mark.parametrize("value", ["short", "g" * 64])
def test_sha256_value_rejects_invalid_digest(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _sha256_value(value)
