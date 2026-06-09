import argparse
import json
import logging
import socket
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from runs import require_latest_run_directory
from storage import load_from_jsonl

DATASET_FILENAME = "repo_files"
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"

SIDEBAR_FIELDS = (
    "repo_name",
    "commit_hash",
    "filename",
    "new_path",
    "change_type",
    "nloc",
    "added_lines",
    "deleted_lines",
)


def _build_sidebar_payload(records: list[dict]) -> list[dict]:
    sidebar_records = []
    for index, record in enumerate(records):
        entry = {"index": index}
        for field in SIDEBAR_FIELDS:
            entry[field] = record.get(field)
        entry["comment_count"] = len(record.get("comments") or [])
        sidebar_records.append(entry)
    return sidebar_records


def _make_handler(
    html_bytes: bytes, records: list[dict], sidebar_payload: list[dict]
):
    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logging.info("%s - %s", self.address_string(), format % args)

        def _send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)

            if path in ("/", "/index.html"):
                self._send_html()
                return

            if path == "/records":
                self._send_json(sidebar_payload)
                return

            if path.startswith("/records/"):
                try:
                    index = int(path[len("/records/") :])
                except ValueError:
                    self._send_json({"error": "invalid index"}, HTTPStatus.BAD_REQUEST)
                    return
                if not (0 <= index < len(records)):
                    self._send_json({"error": "out of range"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json(records[index])
                return

            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return DashboardHandler


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def serve(run_dir: Path, port: int | None, open_browser: bool) -> None:
    records = load_from_jsonl(run_dir, DATASET_FILENAME)
    sidebar_payload = _build_sidebar_payload(records)
    html_bytes = DASHBOARD_HTML.read_bytes()

    chosen_port = port if port is not None else _pick_free_port()
    handler_class = _make_handler(html_bytes, records, sidebar_payload)
    server = ThreadingHTTPServer(("127.0.0.1", chosen_port), handler_class)
    url = f"http://127.0.0.1:{chosen_port}/"

    logging.info("Serving %d records from %s", len(records), run_dir)
    logging.info("Dashboard: %s", url)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as error:
            logging.warning("Could not open browser: %s", error)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        server.server_close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview repo_files.jsonl in a local HTML dashboard."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Run directory containing repo_files.jsonl (defaults to latest run).",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind (default: random free port)."
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the dashboard in the browser.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = _parse_args(argv)
    run_dir = args.run_dir if args.run_dir is not None else require_latest_run_directory()
    serve(run_dir, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(sys.argv[1:])
