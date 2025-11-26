"""Convert PDFs under a data directory into per-page images.

Usage:
    python scripts/pdf_to_images.py --root data --dpi 200 --fmt png

Each immediate child directory A inside root is processed and the images are
written to sibling directory A_img. Loose PDF files at the root get their own
<pdf-stem>_img directory. The original PDF stem is preserved with a suffix
`_page_<id>` so downstream tooling can track provenance easily.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import subprocess


logger = logging.getLogger(__name__)
FORMAT_CONFIG = {
    "png": ("-png", ".png"),
    "jpeg": ("-jpeg", ".jpg"),
    "tiff": ("-tiff", ".tif"),
}
FORMAT_ALIASES = {"jpg": "jpeg"}


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def get_page_count(pdf_path: Path) -> int:
    try:
        result = _run_cmd(["pdfinfo", str(pdf_path)])
    except FileNotFoundError as exc:
        raise RuntimeError("pdfinfo command is required but not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to inspect {pdf_path}: {exc.stderr}") from exc

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1])
            except ValueError as exc:
                raise RuntimeError(f"Unexpected pdfinfo output for {pdf_path}") from exc
    raise RuntimeError(f"Could not determine page count for {pdf_path}")


def convert_pdf(pdf_path: Path, output_dir: Path, fmt_key: str, dpi: int) -> None:
    """Convert a single PDF into images saved inside output_dir."""
    fmt_flag, ext = FORMAT_CONFIG[fmt_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    page_count = get_page_count(pdf_path)
    logger.info("Converting %s (%d pages) -> %s", pdf_path, page_count, output_dir)

    page_files = [
        output_dir / f"{pdf_path.stem}_page_{page:03d}{ext}"
        for page in range(1, page_count + 1)
    ]
    if all(p.exists() for p in page_files):
        logger.info("Skipping %s; %d pages already exist", pdf_path, page_count)
        return

    for page, out_file in enumerate(page_files, start=1):
        if out_file.exists():
            logger.debug("Skipping existing page %d for %s", page, pdf_path.name)
            continue
        prefix = out_file.with_suffix("")
        cmd = [
            "pdftoppm",
            "-r",
            str(dpi),
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            fmt_flag,
        ]
        cmd.extend([str(pdf_path), str(prefix)])
        try:
            _run_cmd(cmd)
        except FileNotFoundError as exc:
            raise RuntimeError("pdftoppm command is required but not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed converting %s page %d: %s", pdf_path, page, exc.stderr.strip()
            )
            continue


def iter_targets(root: Path) -> Iterable[Path]:
    """Yield directories and PDF files inside root."""
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
        elif entry.is_file() and entry.suffix.lower() == ".pdf":
            yield entry


def process_root(root: Path, fmt_key: str, dpi: int) -> None:
    for target in iter_targets(root):
        if target.is_dir():
            output_dir = target.parent / f"{target.name}_img"
            for pdf_file in sorted(target.glob("*.pdf")):
                convert_pdf(pdf_file, output_dir, fmt_key, dpi)
        else:
            output_dir = target.parent / f"{target.stem}_img"
            convert_pdf(target, output_dir, fmt_key, dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDFs to images.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Directory containing PDF folders/files (default: data)",
    )
    parser.add_argument(
        "--fmt",
        default="png",
        help="Image format/extension to write (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Image resolution (default: 200 DPI)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    fmt_key = FORMAT_ALIASES.get(args.fmt.lower(), args.fmt.lower())
    if fmt_key not in FORMAT_CONFIG:
        supported = ", ".join(sorted(FORMAT_CONFIG))
        raise SystemExit(f"Unsupported format '{args.fmt}'. Choose from: {supported}")
    process_root(root, fmt_key=fmt_key, dpi=args.dpi)


if __name__ == "__main__":
    main()
