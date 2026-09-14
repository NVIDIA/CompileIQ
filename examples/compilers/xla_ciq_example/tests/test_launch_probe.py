from __future__ import annotations

from xla_ciq.search.launch_probe import candidate_arg_nbytes


def test_candidates_grow_past_kernel_footprint():
    # A dumped kernel indexes its own compiled shape, so a tiny shape token
    # must still be probed with buffers covering that footprint.
    assert candidate_arg_nbytes(shape_nbytes=8, footprint_nbytes=1024) == [2048, 8192]


def test_candidates_never_shrink_below_shape():
    assert candidate_arg_nbytes(shape_nbytes=4096, footprint_nbytes=16) == [8192, 32768]


def test_candidates_have_headroom_without_a_known_footprint():
    # An exact fit can pass one probe launch and still fault later, so the
    # smallest candidate already covers more than the requested shape.
    assert candidate_arg_nbytes(shape_nbytes=256, footprint_nbytes=0) == [512, 2048]


def test_candidates_are_empty_without_any_size():
    assert candidate_arg_nbytes(shape_nbytes=0, footprint_nbytes=0) == []
