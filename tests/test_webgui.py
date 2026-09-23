"""Security checks for the loopback web interface."""

from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest import TestCase
from unittest.mock import patch

from turkish_markov import webgui


@contextmanager
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), webgui.RequestHandler)
    worker = Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


def call(httpd, method="POST", path="/api/build", *, host=None, origin=None,
         token=None, body="{}", content_type="application/json"):
    connection = HTTPConnection("127.0.0.1", httpd.server_port, timeout=3)
    headers = {"Host": host or f"127.0.0.1:{httpd.server_port}"}
    if origin is not None:
        headers["Origin"] = origin
    if token is not None:
        headers["X-Session-Key"] = token
    if method == "POST":
        headers["Content-Type"] = content_type
    connection.request(method, path, body=body if method == "POST" else None, headers=headers)
    response = connection.getresponse()
    result = response.status, response.read().decode("utf-8")
    connection.close()
    return result


class WebGuiSecurityTests(TestCase):
    def test_build_requires_trusted_host_origin_session_and_json(self) -> None:
        with server() as httpd, patch.object(webgui, "_build") as build:
            origin = f"http://127.0.0.1:{httpd.server_port}"
            self.assertEqual(call(httpd, host="attacker.invalid", origin=origin,
                                  token=webgui.SESSION_KEY, body="")[0], 403)
            self.assertEqual(call(httpd, origin="http://attacker.invalid",
                                  token=webgui.SESSION_KEY, body="")[0], 403)
            self.assertEqual(call(httpd, origin=origin, body="")[0], 403)
            self.assertEqual(call(httpd, origin=origin, token=webgui.SESSION_KEY,
                                  content_type="text/plain")[0], 400)
            self.assertEqual(call(httpd, origin=origin, token=webgui.SESSION_KEY,
                                  body="[]")[0], 400)
            self.assertEqual(call(httpd, origin=origin, token=webgui.SESSION_KEY)[0], 202)
            self.assertEqual(build.call_count, 1)

    def test_shutdown_requires_session_and_trusted_origin(self) -> None:
        with server() as httpd:
            origin = f"http://127.0.0.1:{httpd.server_port}"
            self.assertEqual(call(httpd, path="/api/shutdown", origin=origin, body="")[0], 403)
            self.assertEqual(call(httpd, path="/api/shutdown", origin="http://attacker.invalid",
                                  token=webgui.SESSION_KEY, body="")[0], 403)
            self.assertEqual(call(httpd, method="GET", path="/", host="attacker.invalid")[0], 403)
            status, page = call(httpd, method="GET", path="/")
            self.assertEqual(status, 200)
            self.assertIn(webgui.SESSION_KEY, page)

    def test_overwrite_only_replaces_model_files_in_safe_directories(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.db"
            model.write_bytes(b"database")
            other = root.parent / "other.db"
            nested = root / ".git" / "other.db"
            (root / ".git").mkdir()
            with patch.object(webgui, "PROJECT_ROOT", root):
                webgui._validate_overwrite_target(model)
                for unsafe in (other, nested, root / "notes.txt"):
                    with self.assertRaises(ValueError):
                        webgui._validate_overwrite_target(unsafe)
                with self.assertRaises(ValueError):
                    webgui._build({"input": str(root), "db": str(model), "overwrite": "false"})
                with self.assertRaises(ValueError):
                    webgui._build({"input": str(root), "db": str(other), "overwrite": True})
            self.assertEqual(model.read_bytes(), b"database")
