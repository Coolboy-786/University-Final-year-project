"""Download raw images from registered public dataset sources.

Auth-free sources (HAM10000 from Harvard Dataverse, ISIC via REST API) are
fully implemented.  Auth-required sources (MSLD, DermNet, KaggleAcne,
NormalSkin) raise NotImplementedError until handled in a follow-up.

Downloaded files land in ``data/raw/<source_name>/``.  Each download is
logged to ``data/raw/<source_name>/manifest.csv`` (url, filename, sha256,
timestamp).  Files that already exist with a matching checksum are skipped.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.data.sources import SOURCES, DataSource
from src.utils.paths import get_data_root

logger = logging.getLogger(__name__)

USER_AGENT = "skin-disease-research/0.1"
DATAVERSE_API = "https://dataverse.harvard.edu/api"
HAM10000_DOI = "doi:10.7910/DVN/DBW86T"
ISIC_API = "https://api.isic-archive.com/api/v2"
ISIC_PAGE_SIZE = 200

MANIFEST_COLUMNS: tuple[str, ...] = ("url", "filename", "sha256", "timestamp")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.
    chunk_size : int
        Read chunk size in bytes.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load manifest.csv and return a ``{filename: sha256}`` dict.

    Returns an empty dict if the file does not exist or is empty.
    """
    if not manifest_path.exists():
        return {}
    result: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "filename" in row and "sha256" in row:
                result[row["filename"]] = row["sha256"]
    return result


def _append_manifest(
    manifest_path: Path,
    url: str,
    filename: str,
    sha256: str,
    timestamp: str,
) -> None:
    """Append one row to manifest.csv, writing the header if the file is new."""
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(
            {"url": url, "filename": filename, "sha256": sha256, "timestamp": timestamp}
        )


def _is_cached(path: Path, manifest: dict[str, str]) -> bool:
    """Return True if *path* exists and its sha256 matches the manifest entry."""
    if not path.exists():
        return False
    recorded = manifest.get(path.name)
    if recorded is None:
        return False
    return compute_sha256(path) == recorded


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _http_get_stream(
    session: requests.Session, url: str, dest: Path, desc: str = ""
) -> None:
    """Stream-download *url* to *dest*, using a ``.part`` suffix during transfer.

    Retried up to 5 times with exponential back-off (2 s → 60 s).
    The partial file is deleted on any error so a re-try starts clean.
    """
    tmp = dest.with_suffix(".part")
    try:
        resp = session.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None
        with tmp.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc or dest.name,
            leave=False,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _fetch_file(
    session: requests.Session,
    url: str,
    dest: Path,
    manifest_path: Path,
) -> bool:
    """Download *url* to *dest*; skip entirely if the file is already cached.

    Parameters
    ----------
    session : requests.Session
        Shared HTTP session (carries User-Agent header).
    url : str
        Remote URL to fetch.
    dest : Path
        Local destination path.
    manifest_path : Path
        Path to the source's ``manifest.csv``.

    Returns
    -------
    bool
        ``True`` if the file was downloaded, ``False`` if the cache was hit.
    """
    manifest = _load_manifest(manifest_path)
    if _is_cached(dest, manifest):
        logger.debug("Cache hit: %s", dest.name)
        return False

    _http_get_stream(session, url, dest, desc=dest.name)
    sha = compute_sha256(dest)
    _append_manifest(
        manifest_path,
        url,
        dest.name,
        sha,
        datetime.now(timezone.utc).isoformat(),
    )
    return True


# ---------------------------------------------------------------------------
# Source-specific downloaders
# ---------------------------------------------------------------------------


def _download_ham10000(dest_dir: Path, session: requests.Session, force: bool = False) -> None:
    """Download HAM10000 archive files from Harvard Dataverse."""
    manifest_path = dest_dir / "manifest.csv"

    list_url = (
        f"{DATAVERSE_API}/datasets/:persistentId/versions/:latest/files"
        f"?persistentId={HAM10000_DOI}"
    )
    logger.info("Fetching HAM10000 file list from Dataverse …")
    resp = session.get(list_url, timeout=30)
    resp.raise_for_status()
    file_entries: list[dict[str, object]] = resp.json().get("data", [])

    for entry in tqdm(file_entries, desc="HAM10000"):
        df = entry.get("dataFile") if isinstance(entry, dict) else None
        if not isinstance(df, dict):
            continue
        file_id = df.get("id")
        filename = df.get("filename", "")
        if not file_id or not filename:
            continue

        dest = dest_dir / str(filename)
        if not force and _is_cached(dest, _load_manifest(manifest_path)):
            logger.debug("Cache hit: %s", filename)
            continue

        download_url = f"{DATAVERSE_API}/access/datafile/{file_id}"
        _fetch_file(session, download_url, dest, manifest_path)


def _download_isic(dest_dir: Path, session: requests.Session, force: bool = False) -> None:
    """Download images from the ISIC Archive via their REST API.

    Paginates through all public images.  The manifest is kept in memory
    during the run so cache lookups stay O(1) rather than re-reading the CSV
    for every image.
    """
    manifest_path = dest_dir / "manifest.csv"
    manifest: dict[str, str] = _load_manifest(manifest_path)

    next_url: str | None = f"{ISIC_API}/images/?limit={ISIC_PAGE_SIZE}"
    page = 0
    total_downloaded = 0

    while next_url is not None:
        logger.info("ISIC page %d …", page)
        resp = session.get(next_url, timeout=60)
        resp.raise_for_status()
        data: dict[str, object] = resp.json()

        results = data.get("results", [])
        if not isinstance(results, list):
            break

        for img in tqdm(results, desc=f"ISIC p{page}", leave=False):
            if not isinstance(img, dict):
                continue
            isic_id = img.get("isic_id", "")
            files = img.get("files")
            img_url = (
                files.get("full", {}).get("url", "")
                if isinstance(files, dict)
                else ""
            )
            if not isic_id or not img_url:
                continue

            filename = f"{isic_id}.jpg"
            dest = dest_dir / filename

            if not force and _is_cached(dest, manifest):
                continue

            _http_get_stream(session, str(img_url), dest, desc=filename)
            sha = compute_sha256(dest)
            now = datetime.now(timezone.utc).isoformat()
            _append_manifest(manifest_path, str(img_url), filename, sha, now)
            manifest[filename] = sha
            total_downloaded += 1

        raw_next = data.get("next")
        next_url = str(raw_next) if raw_next else None
        page += 1

    logger.info("ISIC: %d new images downloaded", total_downloaded)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_source(
    entry: DataSource,
    dest_root: Path,
    force: bool = False,
) -> Path:
    """Download one registered source to ``dest_root/<entry.name>/``.

    Parameters
    ----------
    entry : DataSource
        Registry entry to download.
    dest_root : Path
        Parent directory for all raw downloads.
    force : bool
        Re-download even when a cached copy exists.

    Returns
    -------
    Path
        Directory containing the downloaded files.

    Raises
    ------
    NotImplementedError
        For auth-required sources and any source without a registered downloader.
    RuntimeError
        If checksum verification fails (propagated from the stream downloader).
    """
    if entry.requires_auth:
        raise NotImplementedError(
            f"{entry.name!r} requires authentication. "
            "Implement after HAM10000 + ISIC are validated."
        )

    dest_dir = dest_root / entry.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = _make_session()

    if entry.name == "HAM10000":
        _download_ham10000(dest_dir, session, force=force)
    elif entry.name == "ISIC":
        _download_isic(dest_dir, session, force=force)
    else:
        raise NotImplementedError(f"No downloader registered for {entry.name!r}")

    return dest_dir


def download_all(
    dest_root: Path,
    sources: list[DataSource] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download all non-auth-required sources.

    Auth-required sources are skipped with an INFO log; they do not raise.

    Parameters
    ----------
    dest_root : Path
        Parent directory for raw downloads.
    sources : list[DataSource] or None
        Subset to attempt; defaults to all registered sources.
    force : bool
        Re-download even when a cached copy exists.

    Returns
    -------
    dict[str, Path]
        Mapping from source name to downloaded directory (successful only).
    """
    to_download = sources if sources is not None else list(SOURCES.values())
    results: dict[str, Path] = {}

    for entry in to_download:
        if entry.requires_auth:
            logger.info("Skipping auth-required source: %s", entry.name)
            continue
        try:
            results[entry.name] = download_source(entry, dest_root, force=force)
        except Exception:
            logger.exception("Failed to download %s", entry.name)

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.data.download",
        description="Download raw skin-disease images from registered public sources.",
    )
    parser.add_argument(
        "--dest",
        default=str(get_data_root() / "raw"),
        metavar="DIR",
        help="Parent directory for all raw downloads (default: <data_root>/raw).",
    )
    parser.add_argument(
        "--source",
        metavar="NAME",
        action="append",
        dest="sources",
        help=(
            "Download only this source (repeatable). "
            "Choices: %(choices)s. "
            "Defaults to all auth-free sources."
        ),
        choices=list(SOURCES.keys()),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a cached copy with a matching checksum exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered sources and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        print(f"{'NAME':<20} {'AUTH':<6} {'LICENSE'}")
        print("-" * 70)
        for src in SOURCES.values():
            auth = "yes" if src.requires_auth else "no"
            print(f"{src.name:<20} {auth:<6} {src.license}")
        return

    selected: list[DataSource] | None = None
    if args.sources:
        selected = [SOURCES[n] for n in args.sources]

    dest_root = Path(args.dest)
    results = download_all(dest_root, sources=selected, force=args.force)

    print(f"\nDownloaded {len(results)} source(s):")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    _main()
