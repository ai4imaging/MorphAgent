"""Desktop hosting contracts; no model calls or GUI imports."""
import importlib.util
import json
import socket
import sys
import time
from urllib.request import Request, ProxyHandler, build_opener

import pytest

urlopen = build_opener(ProxyHandler({})).open


def runtime_type():
    assert importlib.util.find_spec('morphagent_ui.desktop_runtime'), 'Desktop runtime is missing'
    from morphagent_ui.desktop_runtime import DesktopRuntime
    return DesktopRuntime


@pytest.fixture
def repo(tmp_path):
    site = tmp_path / 'design-preview'
    site.mkdir(parents=True)
    (site / 'index.html').write_text('<html><head></head><body>Workspace</body></html>')
    return tmp_path


def test_owns_ephemeral_loopback_server_and_releases_lock(repo):
    from morphagent_ui.web_service import workspace_lock
    with runtime_type()(repo) as runtime:
        assert runtime.server.server_address[0] == '127.0.0.1'
        port = runtime.server.server_port
        assert port > 0
        assert b'morphagent-token' in urlopen(runtime.url, timeout=3).read()
        request = Request(runtime.url+'/api/bootstrap', headers={'X-MorphAgent-Token':runtime.server.token})
        assert json.load(urlopen(request, timeout=3))['settings']['hasApiKey'] is False
        with pytest.raises(RuntimeError):
            with workspace_lock(repo):
                pass
    assert not runtime.thread.is_alive()
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', port)) != 0
    with workspace_lock(repo):
        pass


def test_failed_bind_releases_workspace_lock(repo):
    from morphagent_ui.web_service import workspace_lock
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        sock.listen()
        with pytest.raises(OSError):
            with runtime_type()(repo, port=sock.getsockname()[1]):
                pass
    with workspace_lock(repo):
        pass


def test_second_launcher_does_not_interrupt_first(repo):
    with runtime_type()(repo) as first:
        with pytest.raises(RuntimeError):
            with runtime_type()(repo):
                pass
        assert urlopen(first.url, timeout=3).status == 200


def test_reference_preview_and_download_require_session_token(repo):
    from urllib.error import HTTPError
    with runtime_type()(repo) as runtime:
        raw=b'Scientific background for this experiment.'
        doc=runtime.service.add_document('expert notes.txt',raw)
        endpoint=runtime.url+'/api/documents/'+doc['id']
        for suffix in ('','/file'):
            with pytest.raises(HTTPError) as error:
                urlopen(endpoint+suffix,timeout=3)
            assert error.value.code==403
        headers={'X-MorphAgent-Token':runtime.server.token}
        preview=json.load(urlopen(Request(endpoint,headers=headers),timeout=3))
        assert preview['text']==raw.decode()
        response=urlopen(Request(endpoint+'/file',headers=headers),timeout=3)
        assert response.read()==raw
        assert response.headers['Content-Disposition'].startswith('attachment;')
        runtime.service.remove_document(doc['id'])
        for suffix in ('','/file'):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(endpoint+suffix,headers=headers),timeout=3)
            assert error.value.code==400


def test_close_stops_owned_process_and_persists_cancelled(repo):
    from morphagent_ui.web_service import ACTIVE
    sample = repo / 'data' / 'dataset' / 'one'
    sample.mkdir(parents=True)
    (sample/'image.tif').write_bytes(b'test-image')
    (repo/'main.py').write_text("import time\nprint('Step 1: Inspect', flush=True)\ntime.sleep(60)\n")
    with runtime_type()(repo) as runtime:
        service = runtime.service
        service.save_settings({'baseUrl':'https://test.invalid/v1','model':'test','apiKey':'fixture-only'})
        dataset = service.add_dataset(str(repo/'data'))
        job = service.start_run({'featureNumber':5,'datasetId':dataset['id'],'question':'test','route':'code','mode':'ultra'})
        deadline = time.monotonic()+5
        while job['id'] not in service.processes and time.monotonic()<deadline:
            time.sleep(.02)
        assert runtime.has_active_runs()
        proc = service.processes[job['id']]
    assert proc.poll() is not None
    assert proc.stdout.closed
    assert service.jobs[job['id']]['status'] == 'cancelled'
    assert not service.processes
    stored = json.loads((repo/'.web_workspace/history.json').read_text())
    assert stored[job['id']]['status'] not in ACTIVE
    # Shutdown must also reject requests already queued by the GUI.
    for operation in (service.start_run, service.start_reuse):
        with pytest.raises(ValueError, match='closing'):
            operation({})


def test_navigation_is_limited_to_exact_local_origin():
    runtime_type()
    from morphagent_ui.desktop_runtime import is_workspace_url
    origin = 'http://127.0.0.1:43210'
    assert is_workspace_url(origin+'/#data', origin)
    assert is_workspace_url(origin+'/api/bootstrap', origin)
    for url in ('http://127.0.0.1:43211/', 'http://localhost:43210/',
                'http://127.0.0.1.evil.test:43210/', 'file:///etc/passwd',
                'javascript:alert(1)', 'https://example.com',
                'http://user@127.0.0.1:43210/', 'http://127.0.0.1:notaport/'):
        assert not is_workspace_url(url, origin)


def test_shutdown_waits_for_result_packaging(repo):
    import threading
    with runtime_type()(repo) as runtime:
        job={'id':'packing','status':'complete','exportStatus':'saving'}
        runtime.service.jobs['packing']=job
        assert runtime.has_active_runs()
        finished=threading.Event()
        def finish():
            job['exportStatus']='ready'
            finished.set()
        timer=threading.Timer(.25,finish)
        timer.start()
        runtime.stop()
        assert finished.is_set()
        timer.join()


def test_stage_parser_does_not_load_either_qt_binding():
    import subprocess
    assert importlib.util.find_spec('morphagent_ui.stages'), 'Qt-free stage parser is missing'
    result = subprocess.run([sys.executable, '-c', '''
import sys
from morphagent_ui.stages import StageDetector
d=StageDetector()
assert d.feed('Step 4: Quantify') == 3
assert not any(k.startswith(('PyQt', 'PySide', 'qtpy')) for k in sys.modules)
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
