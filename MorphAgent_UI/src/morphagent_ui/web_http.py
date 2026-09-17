"""Authenticated loopback HTTP boundary for the browser UI (no external server)."""
from __future__ import annotations

import json
import math
import mimetypes
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from .web_service import WorkspaceService, safe_relative


def json_bytes(value):
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(v) for v in item]
        return item
    return json.dumps(clean(value), ensure_ascii=False, default=str, allow_nan=False).encode('utf-8')


def create_server(service: WorkspaceService, port=8766):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Never log request headers, posted settings, uploaded text, or tokens.
            pass

        def reply(self, status, data, content_type='application/json', filename=None):
            if not isinstance(data, bytes):
                data = json_bytes(data)
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                from urllib.parse import quote
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(filename))
            self.end_headers()
            self.wfile.write(data)

        def dispatch(self):
            try:
                parsed = urlsplit(self.path)
                path = unquote(parsed.path)
                hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
                if self.headers.get('Host') not in hosts:
                    return self.reply(403, {'error': 'Invalid Host.'})
                origin = self.headers.get('Origin')
                if origin and origin not in {'http://' + h for h in hosts}:
                    return self.reply(403, {'error': 'Cross-origin access is not allowed.'})
                if not path.startswith('/api/'):
                    if self.command != 'GET':
                        return self.reply(405, {'error': 'Method not allowed.'})
                    site = service.repo / 'design-preview'
                    relative = safe_relative(path.lstrip('/') or 'index.html')
                    file = (site / relative).resolve()
                    if not file.is_relative_to(site.resolve()) or not file.is_file() or file.suffix not in {'.html', '.css', '.js', '.png', '.jpg', '.svg'}:
                        return self.reply(404, {'error': 'Not found.'})
                    data = file.read_bytes()
                    if relative.as_posix() == 'index.html':
                        data = data.replace(b'</head>', f'<meta name="morphagent-token" content="{self.server.token}"></head>'.encode())
                    return self.reply(200, data, mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                if not secrets.compare_digest(self.headers.get('X-MorphAgent-Token', ''), self.server.token):
                    return self.reply(403, {'error': 'Reload the workspace to start a new local session.'})
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                parts = path.strip('/').split('/')[1:]
                length = int(self.headers.get('Content-Length', '0'))
                if length < 0 or length > 128 * 1024 * 1024:
                    return self.reply(413, {'error': 'Request too large.'})
                raw = self.rfile.read(length) if length else b''
                body = json.loads(raw) if raw and self.headers.get('Content-Type', '').startswith('application/json') else {}
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object.')
                result = None
                method = self.command
                if method == 'GET' and parts == ['bootstrap']:
                    result = service.bootstrap()
                elif method == 'POST' and parts == ['help', 'ask']:
                    result = service.ask_help(body)
                elif method == 'POST' and parts == ['settings']:
                    result = service.save_settings(body)
                elif method == 'POST' and parts == ['datasets']:
                    result = service.add_dataset(body.get('path', ''))
                elif method == 'POST' and parts == ['datasets', 'demo']:
                    result = service.add_dataset(str(service.repo / 'demo/data'))
                elif method == 'POST' and parts == ['imports']:
                    result = service.create_dataset_import()
                elif method == 'PUT' and len(parts) == 2 and parts[0] == 'imports':
                    result = service.put_dataset_file(parts[1], query.get('name', ''), raw)
                elif method == 'POST' and len(parts) == 3 and parts[0] == 'imports' and parts[2] == 'finish':
                    result = service.finish_dataset_import(parts[1])
                elif method == 'POST' and parts == ['documents']:
                    result = service.add_document(query.get('name', ''), raw)
                elif method == 'GET' and len(parts) == 2 and parts[0] == 'documents':
                    result = service.document_preview(parts[1])
                elif method == 'GET' and len(parts) == 3 and parts[0] == 'documents' and parts[2] == 'file':
                    file = service.document_file(parts[1])
                    return self.reply(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'application/octet-stream', file.name)
                elif method == 'POST' and len(parts) == 3 and parts[0] == 'documents' and parts[2] == 'remove':
                    result = service.remove_document(parts[1])
                elif method == 'POST' and parts == ['preflight']:
                    result = service.preflight(body)
                elif method == 'POST' and parts == ['runs']:
                    result = service.start_run(body)
                elif method == 'POST' and parts == ['compute', 'preflight']:
                    result = service.preflight_compute(body)
                elif method == 'POST' and parts in (['compute'], ['reuse']):
                    result = service.start_reuse(body)
                elif method == 'POST' and parts == ['runs', 'load']:
                    result = service.add_existing_run(body.get('path', ''))
                elif method == 'POST' and parts == ['runs', 'demo']:
                    result = service.add_existing_run(str(service.repo / 'demo/data/results/completed_demo_run'))
                elif len(parts) >= 2 and parts[0] == 'runs':
                    key = parts[1]
                    if method == 'GET' and len(parts) == 2:
                        result = service.run_detail(key)
                    elif method == 'POST' and parts[2:] == ['cancel']:
                        result = service.cancel_run(key)
                    elif method == 'POST' and parts[2:] == ['remove']:
                        result = service.remove_run(key)
                    elif method == 'GET' and parts[2:] == ['removal']:
                        result = service.run_removal_plan(key)
                    elif method == 'POST' and parts[2:] == ['export']:
                        result = service.export_results(key)
                    elif method == 'GET' and parts[2:] == ['export']:
                        file = service.export_archive(key)
                        return self.reply(200, file.read_bytes(), 'application/zip', file.name)
                    elif method == 'GET' and parts[2:] == ['logs']:
                        result = service.logs(key, query.get('offset', 0))
                    elif method == 'GET' and parts[2:] == ['evidence']:
                        result = service.feature_evidence(key, query.get('feature', ''))
                    elif method == 'GET' and parts[2:] == ['distribution']:
                        result = service.feature_distribution(key, query.get('feature', ''))
                    elif method == 'GET' and parts[2:] == ['artifact']:
                        file = service.artifact_path(key, query.get('path', ''))
                        if file.stat().st_size > 128 * 1024 * 1024:
                            raise ValueError('Artifact exceeds browser preview limit. Open the results folder locally.')
                        content = file.read_bytes()
                        if file.suffix.lower() in {'.txt', '.csv', '.json'}:
                            content = service.redact(content.decode('utf-8', errors='replace')).encode('utf-8')
                        return self.reply(200, content, mimetypes.guess_type(file.name)[0] or 'application/octet-stream', file.name if query.get('download') else None)
                if result is None:
                    return self.reply(404, {'error': 'Unknown endpoint.'})
                return self.reply(200, result)
            except (ValueError, KeyError, OSError) as exc:
                return self.reply(400, {'error': service.redact(str(exc))})
            except Exception as exc:
                return self.reply(500, {'error': service.redact(str(exc))})

        do_GET = dispatch
        do_POST = dispatch
        do_PUT = dispatch

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.token = secrets.token_urlsafe(32)
    server.daemon_threads = True
    return server
