"""Safe reader for the MATLAB-generated ``.iFLiP3`` container format."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np


_ASSIGNMENT = re.compile(
    r"^header\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*=\s*(.*?)\s*$",
    re.DOTALL,
)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _statements(text: str) -> Iterator[str]:
    """Split MATLAB assignments at semicolons that are outside strings."""

    start = 0
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == "'":
            if in_string and i + 1 < len(text) and text[i + 1] == "'":
                i += 2
                continue
            in_string = not in_string
        elif char == ";" and not in_string:
            yield text[start:i].strip()
            start = i + 1
        i += 1
    if text[start:].strip():
        yield text[start:].strip()


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1].replace("''", "'")
    if _NUMBER.fullmatch(raw):
        value = float(raw)
        return int(value) if value.is_integer() and "." not in raw and "e" not in raw.lower() else value
    if raw.startswith("[") and raw.endswith("]"):
        items = [item for item in re.split(r"[\s,]+", raw[1:-1].strip()) if item]
        if all(_NUMBER.fullmatch(item) for item in items):
            return [_parse_value(item) for item in items]
    raise ValueError(f"Unsupported iFLiP3 header value: {raw!r}")


def _set_nested(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = target
    for part in parts[:-1]:
        existing = node.setdefault(part, {})
        if not isinstance(existing, dict):
            raise ValueError(f"Header path conflict at {path!r}")
        node = existing
    node[parts[-1]] = value


def parse_header(text: str) -> dict[str, Any]:
    """Parse the restricted assignment syntax used in an iFLiP3 header.

    Unlike the vendor MATLAB reader, this function never evaluates header text.
    """

    header: dict[str, Any] = {}
    for statement in _statements(text):
        match = _ASSIGNMENT.fullmatch(statement)
        if not match:
            raise ValueError(f"Unsupported iFLiP3 header statement: {statement!r}")
        _set_nested(header, match.group(1), _parse_value(match.group(2)))
    return header


@dataclass(frozen=True)
class IFLiP3Header(Mapping[str, Any]):
    tree: dict[str, Any]
    raw_text: str

    def __getitem__(self, key: str) -> Any:
        return self.tree[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.tree)

    def __len__(self) -> int:
        return len(self.tree)

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self.tree
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def sampling_frequency(self) -> float:
        return float(self.get_path("state.samplingFreq.Value"))

    @property
    def lifetime_resolution(self) -> float:
        return float(self.get_path("acq.LTResolution"))

    @property
    def sync_rate(self) -> float:
        return float(self.get_path("init.syncRate"))

    @property
    def pulse_interval_ns(self) -> float:
        return 1e9 / self.sync_rate

    @property
    def dead_time_seconds(self) -> float:
        return float(self.get_path("init.deadTime", 25.0)) / 1e9


@dataclass(frozen=True)
class IFLiP3Recording:
    path: Path
    header: IFLiP3Header
    data: np.ndarray
    lifetime_time: np.ndarray
    sample_time: np.ndarray
    marks: np.ndarray

    @property
    def n_channels(self) -> int:
        return self.data.shape[2]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]


def read_iflip3_bytes(
    raw: bytes,
    *,
    source: str | Path = "<memory>",
) -> IFLiP3Recording:
    """Decode iFLiP2/iFLiP3 bytes without executing the MATLAB header."""

    path = Path(source)
    marker = re.search(br"(?m)^header_end\r?\n", raw)
    if marker is None:
        raise ValueError(f"{path} has no header_end marker")

    try:
        header_text = raw[: marker.start()].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} has a non-UTF-8 header") from exc

    header = IFLiP3Header(parse_header(header_text), header_text)
    payload = raw[marker.end() :]
    if len(payload) % 4:
        raise ValueError(f"{path} payload is not a whole number of uint32 values")

    n_channels = int(header.get_path("acq.nChannels"))
    n_bins = int(header.get_path("acq.LTCurveLength"))
    rows_with_marks = n_bins + 1
    values = np.frombuffer(payload, dtype="<u4")
    values_per_sample = rows_with_marks * n_channels
    if values.size % values_per_sample:
        raise ValueError(
            f"{path} has {values.size} values; expected a multiple of {values_per_sample}"
        )
    n_samples = values.size // values_per_sample

    # MATLAB wrote the reshape in column-major order as [bin, channel, sample].
    cube = values.reshape((rows_with_marks, n_channels, n_samples), order="F")
    cube = np.transpose(cube, (0, 2, 1)).copy()
    marks = cube[-1, :, 0].copy()
    data = cube[:-1]

    lifetime_time = np.arange(n_bins, dtype=float) * header.lifetime_resolution
    sample_time = (np.arange(n_samples, dtype=float) + 0.5) / header.sampling_frequency
    return IFLiP3Recording(path, header, data, lifetime_time, sample_time, marks)


def read_iflip3(path: str | Path) -> IFLiP3Recording:
    """Read an iFLiP2/iFLiP3 file without executing its MATLAB header."""

    path = Path(path)
    return read_iflip3_bytes(path.read_bytes(), source=path)
