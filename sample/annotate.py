import argparse
import json
import logging
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from sample.sample import SAMPLE_FILENAME, SAMPLES_DIRECTORY_NAME
from storage import append_to_jsonl, iter_from_jsonl, truncate_broken_tail
from storage.runs import resolve_dataset_and_run
from storage.timestamped_dirs import (
    find_latest_timestamped_directory,
    resolve_timestamped_argument,
)

ANNOTATIONS_FILENAME = "annotations"
PAGE_PATH = Path(__file__).with_name("annotate.html")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _sample_directory(run_dir: Path, sample_dir_argument: str | None) -> Path:
    samples_root = run_dir / SAMPLES_DIRECTORY_NAME
    if sample_dir_argument:
        return resolve_timestamped_argument(
            sample_dir_argument, samples_root, label="sample directory"
        )
    latest_directory = find_latest_timestamped_directory(samples_root)
    if latest_directory is None:
        raise SystemExit(
            f"No sample found under {samples_root}. Run python -m sample.sample first."
        )
    return latest_directory


def _latest_annotations(sample_dir: Path) -> dict[str, dict]:
    """The log is append-only, so the last line for an id is its current state."""
    if not (sample_dir / f"{ANNOTATIONS_FILENAME}.jsonl").exists():
        return {}
    return {
        annotation["id"]: annotation
        for annotation in iter_from_jsonl(sample_dir, ANNOTATIONS_FILENAME)
    }


def _clean_tags(tags) -> list[str]:
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("tags must be a list of strings")
    cleaned_tags = []
    for tag in tags:
        cleaned_tag = "-".join(tag.strip().lower().split())
        if cleaned_tag and cleaned_tag not in cleaned_tags:
            cleaned_tags.append(cleaned_tag)
    return cleaned_tags


def _annotation_from_payload(payload, item_ids: set[str]) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("annotation must be a JSON object")
    item_id = payload.get("id")
    if item_id not in item_ids:
        raise ValueError(f"unknown item id: {item_id!r}")
    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")
    return {
        "id": item_id,
        "tags": _clean_tags(payload.get("tags", [])),
        "notes": notes,
        "done": bool(payload.get("done", False)),
        "revealed": bool(payload.get("revealed", False)),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _make_handler(sample_dir: Path):
    items = list(iter_from_jsonl(sample_dir, SAMPLE_FILENAME))
    item_ids = {item["id"] for item in items}
    manifest_path = sample_dir / f"{SAMPLE_FILENAME}.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    class AnnotationHandler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, data) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            if self.path == "/":
                page = PAGE_PATH.read_bytes()
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path == "/api/items":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "sample_dir": str(sample_dir),
                        "manifest": manifest,
                        "items": items,
                        "annotations": _latest_annotations(sample_dir),
                    },
                )
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/annotations":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                annotation = _annotation_from_payload(payload, item_ids)
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            append_to_jsonl([annotation], sample_dir, ANNOTATIONS_FILENAME)
            self._send_json(HTTPStatus.OK, annotation)

        def log_message(self, format, *args):
            logging.debug(format, *args)

    return AnnotationHandler, len(items)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory of the sample (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory of the sample (defaults to the latest run in the dataset)",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help="Sample directory to annotate (defaults to the latest sample in the run)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to serve on (default: {DEFAULT_PORT})",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    _, run_directory = resolve_dataset_and_run(args.dataset_dir, args.run_dir)
    sample_directory = _sample_directory(run_directory, args.sample_dir)
    # A save killed mid-write leaves a partial last line that would break loading.
    removed_bytes = truncate_broken_tail(sample_directory, ANNOTATIONS_FILENAME)
    if removed_bytes:
        logging.warning("Dropped %d bytes of a partially written annotation", removed_bytes)

    handler, item_count = _make_handler(sample_directory)
    server = HTTPServer((DEFAULT_HOST, args.port), handler)
    logging.info("Annotating %d items in %s", item_count, sample_directory)
    logging.info("Open http://%s:%d", DEFAULT_HOST, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
