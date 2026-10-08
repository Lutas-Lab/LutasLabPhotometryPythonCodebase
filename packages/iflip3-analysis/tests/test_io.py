import numpy as np
from iflip3.io import parse_header, read_iflip3_bytes


def test_parse_header_handles_multiline_string_and_semicolon():
    text = (
        "header.state.samplingFreq.Value = 10;\n"
        "header.state.hardwareInfo.Text = 'line one\nPart# 1; S/N 2';\n"
        "header.acq.LTResolution = 0.1;\n"
    )
    parsed = parse_header(text)
    assert parsed["state"]["samplingFreq"]["Value"] == 10
    assert parsed["state"]["hardwareInfo"]["Text"] == "line one\nPart# 1; S/N 2"
    assert parsed["acq"]["LTResolution"] == 0.1


def test_read_synthetic_file_bytes():
    header = (
        b"header.state.samplingFreq.Value = 2;\n"
        b"header.acq.LTResolution = 0.5;\n"
        b"header.acq.nChannels = 1;\n"
        b"header.acq.LTCurveLength = 2;\n"
        b"header.init.syncRate = 80000000;\n"
        b"header.init.deadTime = 25;\n"
        b"header_end\n"
    )
    # Two samples. Each has two lifetime bins followed by a marker.
    values = np.array([1, 2, 9, 3, 4, 8], dtype="<u4")
    recording = read_iflip3_bytes(
        header + values.tobytes(),
        source="tiny.iFLiP3",
    )
    np.testing.assert_array_equal(recording.data[:, :, 0], [[1, 3], [2, 4]])
    np.testing.assert_array_equal(recording.marks, [9, 8])
    np.testing.assert_allclose(recording.lifetime_time, [0.0, 0.5])
    np.testing.assert_allclose(recording.sample_time, [0.25, 0.75])
