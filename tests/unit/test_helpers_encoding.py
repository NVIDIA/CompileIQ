"""
Tests for encoding/decoding utilities in compileiq/utils/helpers.py.
"""

from compileiq.utils.helpers import _encode_for_core, _decode_from_core


# ---------------------------------------------------------------------------
# _encode_for_core / _decode_from_core — base64 round-trip
# ---------------------------------------------------------------------------


class TestCoreEncoding:
    """These functions encode parameter keys to base64 so the core binary
    can use them as identifiers.  The round-trip must be lossless for
    every string a user might use as a parameter name."""

    def test_plain_ascii_round_trip(self):
        assert _decode_from_core(_encode_for_core("learning_rate")) == "learning_rate"

    def test_unicode_round_trip(self):
        """Users might have non-ASCII parameter names (e.g. from localized configs)."""
        assert _decode_from_core(_encode_for_core("lr_\u03b1")) == "lr_\u03b1"

    def test_empty_string_round_trip(self):
        assert _decode_from_core(_encode_for_core("")) == ""

    def test_special_characters_round_trip(self):
        """Dots, slashes, and spaces are common in path-like parameter names."""
        for s in ["a/b", "x.y", "with spaces", 'has"quotes', "key=value"]:
            assert _decode_from_core(_encode_for_core(s)) == s

    def test_encoded_output_is_valid_base64(self):
        """The encoded string must only contain base64-safe characters,
        since the core binary uses it as an identifier."""
        import base64

        encoded = _encode_for_core("test_key")
        # This should not raise — if it does, the encoding is broken
        base64.b64decode(encoded)
