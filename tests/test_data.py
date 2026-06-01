"""Tests for src/data/sources.py and src/data/download.py."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.download import (
    _append_manifest,
    _fetch_file,
    _is_cached,
    _load_manifest,
    compute_sha256,
)
from src.data.sources import (
    CANONICAL_CLASSES,
    CLASS_TO_SOURCES,
    SOURCES,
    DataSource,
    get_source,
    sources_for_class,
)


# ---------------------------------------------------------------------------
# sources.py
# ---------------------------------------------------------------------------


class TestCanonicalClasses:
    def test_count(self) -> None:
        assert len(CANONICAL_CLASSES) == 23

    def test_no_duplicates(self) -> None:
        assert len(CANONICAL_CLASSES) == len(set(CANONICAL_CLASSES))


class TestSourcesRegistry:
    def test_required_sources_present(self) -> None:
        for name in ("HAM10000", "ISIC", "MSLD", "DermNet", "KaggleAcne", "NormalSkin"):
            assert name in SOURCES, f"Missing source: {name}"

    def test_auth_free_sources(self) -> None:
        assert not SOURCES["HAM10000"].requires_auth
        assert not SOURCES["ISIC"].requires_auth

    def test_auth_required_sources(self) -> None:
        for name in ("MSLD", "DermNet", "KaggleAcne", "NormalSkin"):
            assert SOURCES[name].requires_auth, f"{name} should require auth"

    def test_label_mapping_values_are_canonical(self) -> None:
        canonical_set = set(CANONICAL_CLASSES)
        for src in SOURCES.values():
            for canonical in src.label_mapping.values():
                assert canonical in canonical_set, (
                    f"{src.name}: label_mapping value {canonical!r} not in CANONICAL_CLASSES"
                )

    def test_native_labels_superset_of_mapping_keys(self) -> None:
        for src in SOURCES.values():
            for key in src.label_mapping:
                assert key in src.native_labels, (
                    f"{src.name}: mapping key {key!r} missing from native_labels"
                )

    def test_get_source_returns_correct_entry(self) -> None:
        entry = get_source("HAM10000")
        assert entry.name == "HAM10000"

    def test_get_source_raises_for_unknown(self) -> None:
        with pytest.raises(KeyError):
            get_source("NonExistentSource")


class TestClassToSources:
    def test_not_empty(self) -> None:
        assert CLASS_TO_SOURCES

    def test_ham10000_covers_expected_classes(self) -> None:
        for cls in ("melanoma", "melanocytic-nevi", "basal-cell-carcinoma", "actinic-keratosis"):
            assert "HAM10000" in CLASS_TO_SOURCES.get(cls, []), (
                f"HAM10000 missing from CLASS_TO_SOURCES[{cls!r}]"
            )

    def test_no_duplicate_source_per_class(self) -> None:
        for cls, srcs in CLASS_TO_SOURCES.items():
            assert len(srcs) == len(set(srcs)), f"Duplicate source entries for class {cls!r}"

    def test_all_values_are_registered_source_names(self) -> None:
        for cls, srcs in CLASS_TO_SOURCES.items():
            for name in srcs:
                assert name in SOURCES, f"Unknown source name {name!r} in CLASS_TO_SOURCES[{cls!r}]"

    def test_sources_for_class_roundtrip(self) -> None:
        entries = sources_for_class("melanoma")
        names = [e.name for e in entries]
        assert "HAM10000" in names
        assert "ISIC" in names

    def test_sources_for_class_empty_for_unknown(self) -> None:
        assert sources_for_class("nonexistent-class") == []


# ---------------------------------------------------------------------------
# download.py — compute_sha256
# ---------------------------------------------------------------------------


class TestComputeSha256:
    def test_known_value(self, tmp_path: Path) -> None:
        content = b"hello world"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_sha256(f) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_sha256(f) == expected

    def test_large_file_chunked(self, tmp_path: Path) -> None:
        content = b"x" * (3 * 1024 * 1024)  # 3 MB, spans multiple 1 MB chunks
        f = tmp_path / "big.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_sha256(f, chunk_size=1 << 20) == expected


# ---------------------------------------------------------------------------
# download.py — manifest I/O
# ---------------------------------------------------------------------------


class TestManifest:
    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _load_manifest(tmp_path / "manifest.csv") == {}

    def test_append_creates_file_with_header(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.csv"
        _append_manifest(p, "http://x.com/a.jpg", "a.jpg", "abc", "2024-01-01T00:00:00")
        assert p.exists()
        rows = list(csv.DictReader(p.open()))
        assert rows[0]["filename"] == "a.jpg"
        assert rows[0]["sha256"] == "abc"
        assert rows[0]["url"] == "http://x.com/a.jpg"

    def test_append_multiple_rows(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.csv"
        _append_manifest(p, "http://x.com/a.jpg", "a.jpg", "aaa", "2024-01-01T00:00:00")
        _append_manifest(p, "http://x.com/b.jpg", "b.jpg", "bbb", "2024-01-01T00:00:01")
        rows = list(csv.DictReader(p.open()))
        assert len(rows) == 2
        assert rows[1]["filename"] == "b.jpg"

    def test_load_returns_filename_to_sha256(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.csv"
        _append_manifest(p, "http://x.com/a.jpg", "a.jpg", "deadbeef", "2024-01-01T00:00:00")
        loaded = _load_manifest(p)
        assert loaded == {"a.jpg": "deadbeef"}

    def test_load_roundtrip_multiple_rows(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.csv"
        _append_manifest(p, "http://x.com/a.jpg", "a.jpg", "aaa", "2024-01-01T00:00:00")
        _append_manifest(p, "http://x.com/b.jpg", "b.jpg", "bbb", "2024-01-01T00:00:01")
        loaded = _load_manifest(p)
        assert loaded == {"a.jpg": "aaa", "b.jpg": "bbb"}


# ---------------------------------------------------------------------------
# download.py — _is_cached
# ---------------------------------------------------------------------------


class TestIsCached:
    def test_cached_when_file_exists_and_sha_matches(self, tmp_path: Path) -> None:
        content = b"image bytes"
        sha = hashlib.sha256(content).hexdigest()
        f = tmp_path / "img.jpg"
        f.write_bytes(content)
        assert _is_cached(f, {"img.jpg": sha})

    def test_not_cached_when_file_missing(self, tmp_path: Path) -> None:
        assert not _is_cached(tmp_path / "missing.jpg", {"missing.jpg": "abc"})

    def test_not_cached_when_sha_wrong(self, tmp_path: Path) -> None:
        content = b"image bytes"
        f = tmp_path / "img.jpg"
        f.write_bytes(content)
        assert not _is_cached(f, {"img.jpg": "wrongsha"})

    def test_not_cached_when_not_in_manifest(self, tmp_path: Path) -> None:
        content = b"image bytes"
        f = tmp_path / "img.jpg"
        f.write_bytes(content)
        assert not _is_cached(f, {})


# ---------------------------------------------------------------------------
# download.py — _fetch_file (mocked network)
# ---------------------------------------------------------------------------


class TestFetchFileWithMockedNetwork:
    def test_cache_hit_skips_http(self, tmp_path: Path) -> None:
        """File exists with correct sha256 → _http_get_stream never called."""
        content = b"fake image data"
        sha = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "ISIC_0000001.jpg"
        dest.write_bytes(content)

        manifest_path = tmp_path / "manifest.csv"
        _append_manifest(
            manifest_path, "http://example.com/img.jpg", dest.name, sha, "2024-01-01T00:00:00"
        )

        session = MagicMock()
        with patch("src.data.download._http_get_stream") as mock_stream:
            result = _fetch_file(session, "http://example.com/img.jpg", dest, manifest_path)

        assert result is False
        mock_stream.assert_not_called()

    def test_cache_miss_calls_http_and_writes_manifest(self, tmp_path: Path) -> None:
        """Fresh file → HTTP download called and manifest row written with correct sha256."""
        content = b"fresh image data"
        dest = tmp_path / "ISIC_0000002.jpg"
        manifest_path = tmp_path / "manifest.csv"
        url = "http://example.com/ISIC_0000002.jpg"

        def fake_stream(
            session: MagicMock, url: str, path: Path, desc: str = ""
        ) -> None:
            path.write_bytes(content)

        session = MagicMock()
        with patch("src.data.download._http_get_stream", side_effect=fake_stream) as mock_stream:
            result = _fetch_file(session, url, dest, manifest_path)

        assert result is True
        mock_stream.assert_called_once()

        manifest = _load_manifest(manifest_path)
        expected_sha = hashlib.sha256(content).hexdigest()
        assert manifest.get(dest.name) == expected_sha

    def test_manifest_url_recorded_correctly(self, tmp_path: Path) -> None:
        """manifest.csv url column matches the requested URL."""
        content = b"data"
        dest = tmp_path / "img.jpg"
        manifest_path = tmp_path / "manifest.csv"
        url = "http://example.com/img.jpg"

        def fake_stream(
            session: MagicMock, url: str, path: Path, desc: str = ""
        ) -> None:
            path.write_bytes(content)

        session = MagicMock()
        with patch("src.data.download._http_get_stream", side_effect=fake_stream):
            _fetch_file(session, url, dest, manifest_path)

        rows = list(csv.DictReader(manifest_path.open()))
        assert len(rows) == 1
        assert rows[0]["url"] == url

    def test_second_call_is_cache_hit(self, tmp_path: Path) -> None:
        """After a successful download, calling _fetch_file again is a cache hit."""
        content = b"data"
        dest = tmp_path / "img.jpg"
        manifest_path = tmp_path / "manifest.csv"
        url = "http://example.com/img.jpg"

        def fake_stream(
            session: MagicMock, url: str, path: Path, desc: str = ""
        ) -> None:
            path.write_bytes(content)

        session = MagicMock()
        with patch("src.data.download._http_get_stream", side_effect=fake_stream):
            first = _fetch_file(session, url, dest, manifest_path)

        with patch("src.data.download._http_get_stream") as mock_stream:
            second = _fetch_file(session, url, dest, manifest_path)

        assert first is True
        assert second is False
        mock_stream.assert_not_called()


# ---------------------------------------------------------------------------
# download.py — download_source auth guard
# ---------------------------------------------------------------------------


class TestDownloadSourceAuthGuard:
    def test_auth_required_raises_not_implemented(self, tmp_path: Path) -> None:
        from src.data.download import download_source

        for name in ("MSLD", "DermNet", "KaggleAcne", "NormalSkin"):
            with pytest.raises(NotImplementedError):
                download_source(SOURCES[name], tmp_path)
