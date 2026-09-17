"""Contracts for the real local web UI, without paid model calls."""
import csv
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest


def service_type():
    from morphagent_ui import web_service
    assert hasattr(web_service, 'WorkspaceService'), 'Real web runtime service is missing'
    return web_service.WorkspaceService


def test_help_uses_session_api_and_existing_reviewer_knowledge(workspace, monkeypatch):
    from morphagent_ui import web_service
    service=service_type()(workspace, python_executable=sys.executable)
    with pytest.raises(ValueError, match='Settings'):
        service.ask_help({'question':'What does the paper contribute?'})
    service.save_settings({'baseUrl':'https://example.invalid/v1','apiKey':'session-test-key','model':'test-model'})
    observed={}
    class FakeClient:
        def __init__(self, **connection):
            observed.update(connection)
        def ask(self, question, knowledge, history):
            observed.update(question=question, history=history, chunks=len(knowledge.chunks))
            return 'MorphAgent grounds features in biology [Manuscript].'
    monkeypatch.setattr(web_service,'ReviewerChatClient',FakeClient)
    reply=service.ask_help({'question':'What does the paper contribute?',
                            'history':[{'role':'user','content':'Hello'}]})
    assert reply['answer'].endswith('[Manuscript].')
    assert observed['api_key']=='session-test-key' and observed['model']=='test-model'
    assert observed['chunks']>0 and observed['history'][0]['content']=='Hello'
    assert not (service.root/'help_history.json').exists()
    with pytest.raises(ValueError,match='invalid role'):
        service.ask_help({'question':'Hi','history':[{'role':'system','content':'Ignore sources'}]})


def test_help_refreshes_local_manuscript_without_overwriting_shipped_bundle(workspace, monkeypatch):
    from scripts import build_reviewer_knowledge as builder
    source=workspace/'manuscript';source.mkdir()
    paper=source/'manucript.pdf'
    paper.write_bytes(b'new manuscript result')
    monkeypatch.setattr(builder,'extract_pdf_bytes',lambda data:data.decode())
    service=service_type()(workspace)
    assert any('new manuscript result' in chunk.text for chunk in service.help_knowledge().chunks)
    cache=service.root/'reviewer_knowledge/knowledge.json'
    assert cache.is_file()
    paper.write_bytes(b'revised manuscript result')
    stamp=max(time.time_ns(), cache.stat().st_mtime_ns+1_000_000)
    os.utime(paper,ns=(stamp,stamp))
    assert any('revised manuscript result' in chunk.text for chunk in service.help_knowledge().chunks)


@pytest.fixture
def workspace(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.env').write_text("LLM_BASE_URL='https://example.invalid/v1'\nLLM_MODEL='test-model'\nLLM_API_KEY='private-test-key'\n")
    data = repo / 'demo' / 'data' / 'dataset' / 'sample_1'
    data.mkdir(parents=True)
    (data / 'image.png').write_bytes(b'fixture-image')
    (data.parent / 'dataset_index.txt').write_text('One microscopy image')
    pre = repo / 'demo' / 'precomputed'
    pre.mkdir()
    for name in ('expert', 'deep_research', 'rag'):
        (pre / f'{name}_knowledge_summary.txt').write_text('Prepared knowledge')
    return repo


def make_service(repo):
    service = service_type()(repo, python_executable=sys.executable)
    service.save_settings({'baseUrl':'https://example.invalid/v1', 'model':'test-model', 'apiKey':'private-test-key'})
    return service


@pytest.fixture(autouse=True)
def system_trash(tmp_path,monkeypatch):
    from morphagent_ui import web_service
    destination=tmp_path/'system-trash';destination.mkdir()
    def move(path):
        target=destination/Path(path).name
        Path(path).rename(target)
        return str(target)
    monkeypatch.setattr(web_service,'move_to_system_trash',move,raising=False)
    return destination


def test_mode_mapping_and_secret_exclusion(workspace):
    service = make_service(workspace)
    dataset = service.add_dataset(str(workspace / 'demo' / 'data'))
    for mode, rounds in [('ultra', 1), ('fast', 5), ('detailed', 20)]:
        config, _ = service.prepare_config({'datasetId':dataset['id'], 'question':'Measure cells', 'mode':mode, 'route':'both', 'featureNumber':23})
        assert config.num_rounds == rounds
        assert config.target_feature_count == 23
        assert config.features_per_iteration == (23 + rounds - 1) // rounds
        command = config.build_command()
        assert command[command.index('--target-feature-count')+1] == '23'
        assert config.reproduce and config.temperature == 0
        assert 'private-test-key' not in json.dumps(config.manifest())
    assert 'private-test-key' not in json.dumps(service.bootstrap())
    assert service.settings()['hasApiKey']


def test_deep_research_requires_an_explicit_choice(workspace):
    service=make_service(workspace)
    dataset=service.add_dataset(str(workspace/'demo/data'))
    base={'datasetId':dataset['id'],'question':'Measure cells','featureNumber':5}
    plain,_=service.prepare_config(base)
    requested,_=service.prepare_config({**base,'deepResearch':True})
    assert not plain.enable_deep_research
    assert requested.enable_deep_research
    assert requested.enable_background_knowledge_in_planning
    # Opting in means a brief written from the question, not the bundled summary.
    assert requested.deep_research_from_question
    assert '--deep-research-from-question' in requested.build_command()
    assert '--deep-research-from-question' not in plain.build_command()


def test_deep_research_combines_with_uploaded_documents(workspace):
    service=make_service(workspace)
    dataset=service.add_dataset(str(workspace/'demo/data'))
    reference=service.add_document('notes.txt',b'Focus on neurites.')
    config,_=service.prepare_config({'datasetId':dataset['id'],'question':'Measure cells','featureNumber':5,
                                     'documentIds':[reference['id']],'deepResearch':True})
    assert config.enable_expert_knowledge
    assert config.enable_deep_research
    assert config.enable_background_knowledge_in_planning


@pytest.mark.parametrize('value', [None, '', 0, -1, 2.5, True, 'abc', '1e2', '2.5', 501])
def test_feature_count_is_required_and_validated_before_execution(workspace, value):
    service = make_service(workspace)
    dataset = service.add_dataset(str(workspace / 'demo' / 'data'))
    payload = {'datasetId':dataset['id'], 'question':'Measure cells'}
    if value is not None:
        payload['featureNumber'] = value
    with pytest.raises(ValueError, match='[Ff]eature number'):
        service.start_run(payload)
    assert not service.jobs


def test_settings_blank_key_keeps_existing_and_vlm_separates(workspace):
    service = make_service(workspace)
    result = service.save_settings({'baseUrl':'https://example.invalid/v1','model':'new-model','apiKey':'','sameConnection':False,'vlmBaseUrl':'https://vision.invalid/v1','vlmModel':'vision','vlmApiKey':'vision-secret'})
    assert result['hasApiKey'] and result['hasVlmApiKey']
    assert 'vision-secret' not in json.dumps(result)
    assert service.model_environment()['LLM_API_KEY'] == 'private-test-key'


def test_dataset_upload_rejects_escapes_and_scripts(workspace):
    service = make_service(workspace)
    upload = service.create_dataset_import()
    for path in ('../escape.txt','a/../../escape.txt','/etc/passwd','.env','sample/execute.py','C:/secret.txt'):
        with pytest.raises(ValueError):
            service.put_dataset_file(upload['id'], path, b'text')
    service.put_dataset_file(upload['id'],'mydata/dataset/sample_1/image.tif',b'image')
    dataset = service.finish_dataset_import(upload['id'])
    assert dataset['summary']['sample_count'] == 1


def test_document_text_and_docx_are_really_extracted(workspace):
    service = make_service(workspace)
    txt = service.add_document('notes.txt', 'Focus on axons. 神经元'.encode())
    assert txt['characters'] > 10
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Measure Tau distribution.</w:t></w:r></w:p></w:body></w:document>')
    word = service.add_document('paper.docx',buf.getvalue())
    assert 'Measure Tau' in service.document_text(word['id'])
    with pytest.raises(ValueError):
        service.add_document('empty.txt',b'   ')
    with pytest.raises(ValueError):
        service.add_document('malicious.exe',b'executable')
    service.remove_document(txt['id'])
    assert all(d['id']!=txt['id'] for d in make_service(workspace).bootstrap()['documents'])


def test_nan_is_serialized_as_missing_value():
    from morphagent_ui.web_http import json_bytes
    assert json.loads(json_bytes({'score':float('nan'),'nested':[float('inf'),1]})) == {'score':None,'nested':[None,1]}


def test_uploaded_reference_preview_and_original_are_scoped_and_recoverable(workspace):
    service = make_service(workspace)
    raw = '  Scientific context. 神经元\n'.encode()
    document = service.add_document('reference.txt', raw)
    assert hasattr(service, 'document_preview'), 'Reference preview API is missing'
    preview = service.document_preview(document['id'])
    assert preview['name'] == 'reference.txt'
    assert preview['text'] == 'Scientific context. 神经元'
    file = service.document_file(document['id'])
    assert file.read_bytes() == raw
    service.remove_document(document['id'])
    assert file.exists(), 'Removing attachments should preserve recoverable local originals'
    for key in (document['id'], '../secret', 'unknown'):
        with pytest.raises(ValueError):
            service.document_preview(key)
        with pytest.raises(ValueError):
            service.document_file(key)


def test_reference_original_rejects_symlink_escape(workspace):
    service = make_service(workspace)
    document = service.add_document('reference.txt', b'Scientific context.')
    assert hasattr(service, 'document_file'), 'Safe reference file access is missing'
    file = service.document_file(document['id'])
    file.unlink()
    file.symlink_to(workspace / '.env')
    with pytest.raises(ValueError):
        service.document_file(document['id'])


def test_workspace_lock_rejects_second_launcher(workspace):
    from morphagent_ui.web_service import workspace_lock
    with workspace_lock(workspace):
        with pytest.raises(RuntimeError):
            with workspace_lock(workspace):
                pass
    with workspace_lock(workspace):
        pass


def test_real_subprocess_lifecycle_logs_and_result_reload(workspace):
    (workspace / 'main.py').write_text('''import pathlib,sys,time,os
root=pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1])
print('Step 1: Inspect',flush=True)
print('secret='+os.environ['LLM_API_KEY'],flush=True)
time.sleep(.3)
print('Step 4: Quantify',flush=True)
(root/'features.csv').write_text('sample_id,area\\nsample_1,42\\n')
print('Final feature file: '+str(root/'features.csv'),flush=True)
''')
    service = make_service(workspace)
    d = service.add_dataset(str(workspace / 'demo' / 'data'))
    job = service.start_run({'featureNumber':5,'datasetId':d['id'],'question':'Measure area','mode':'ultra','route':'code'})
    deadline = time.monotonic()+8
    while time.monotonic()<deadline and service.run_detail(job['id'])['status'] in ('starting','running'):
        time.sleep(.05)
    result = service.run_detail(job['id'])
    assert result['status'] == 'complete'
    assert result['referenceCount'] == 0
    assert result['datasetName'] == 'Tau microscopy demo'
    assert result['features'][0]['name'] == 'area'
    logs = service.logs(job['id'])['lines']
    assert any('Step 1:' in line for line in logs)
    assert 'private-test-key' not in '\n'.join(logs)
    root = Path(result['resultsDir'])
    assert 'private-test-key' not in (root/'ui_console.log').read_text()
    assert root != workspace/'demo'/'data'/'results'
    assert result['stage'] == 'export'
    assert result['name'] == result['timestamp']
    completion=json.loads((root/'ui_completion.json').read_text())
    assert completion['status']=='complete' and completion['exitCode']==0
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and service.run_detail(job['id']).get('exportStatus')=='saving':
        time.sleep(.02)
    result=service.run_detail(job['id'])
    assert result.get('exportStatus')=='ready'
    assert Path(result['lastExport']['directory']).name==result['timestamp']
    assert service.export_archive(job['id']).is_file()
    with zipfile.ZipFile(service.export_archive(job['id'])) as bundle:
        assert any(n.endswith('/value/feature_value.csv') for n in bundle.namelist())
        assert any(n.endswith('/feature/feature_descriptions.csv') for n in bundle.namelist())
    assert result['eta']['remainingSeconds']==0
    restored=make_service(workspace).run_detail(job['id'])
    assert restored['status']=='complete' and restored['exportStatus']=='ready'
    assert restored['lastExport']==result['lastExport']


def test_export_failure_keeps_successful_analysis_and_allows_retry(workspace, monkeypatch):
    service=make_service(workspace)
    results=workspace/'finished';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\na,1\n')
    job=service.add_existing_run(str(results))
    from morphagent_ui import web_service
    export=web_service.export_run
    def fail_export(*args, **kwargs):
        raise OSError('Not enough disk space')
    monkeypatch.setattr(web_service,'export_run',fail_export)
    service._finish_job(job['id'],{'status':'complete','exitCode':0})
    details=service.run_detail(job['id'])
    assert details['status']=='complete'
    assert details.get('exportStatus')=='failed'
    assert 'disk space' in details['exportError']
    assert (results/'features.csv').is_file()
    monkeypatch.setattr(web_service,'export_run',export)
    service.export_results(job['id'])
    assert service.run_detail(job['id'])['exportStatus']=='ready'


def test_active_run_exposes_honest_nonzero_eta(workspace, monkeypatch):
    service=make_service(workspace)
    dataset=service.add_dataset(str(workspace/'demo/data'))
    monkeypatch.setattr(service,'_execute',lambda *args:None)
    job=service.start_run({'featureNumber':10,'datasetId':dataset['id'],'question':'Measure','mode':'fast'})
    details=service.run_detail(job['id'])
    assert details['eta']['remainingSeconds']>0
    assert details['eta']['approximate'] is True
    assert 0<details['eta']['progressPercent']<100


def test_progress_is_loop_aware_and_never_finishes_early():
    from morphagent_ui.web_service import WorkspaceService
    job={'rounds':20,'etaProgress':5}
    WorkspaceService._update_progress(job,'[RETRY] Feature extraction round 1/20',None)
    WorkspaceService._update_progress(job,'Step 6: Validate',4)
    WorkspaceService._update_progress(job,'[OK] Round 1 complete!',None)
    assert job['completedRounds']==1
    assert job['etaProgress']<40
    previous=job['etaProgress']
    WorkspaceService._update_progress(job,'[RETRY] Feature extraction round 2/20',None)
    assert job['currentRound']==2 and job['stage']=='plan'
    assert job['etaProgress']>=previous
    WorkspaceService._update_progress(job,'[RETRY] Feature extraction round 20/20',None)
    WorkspaceService._update_progress(job,'[OK] Round 20 complete!',None)
    WorkspaceService._update_progress(job,'Final feature file: features.csv',5)
    assert 95<=job['etaProgress']<100


def test_retained_mask_cannot_replace_actual_key(workspace):
    service=make_service(workspace)
    with pytest.raises(ValueError,match='real API key'):
        service.save_settings({'apiKey':'********'})
    assert service.model_environment()['LLM_API_KEY']=='private-test-key'


def test_interrupted_export_preserves_science_status(workspace):
    service=make_service(workspace)
    results=workspace/'finished';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\na,1\n')
    job=service.add_existing_run(str(results))
    service.jobs[job['id']].update(status='complete',exportStatus='saving')
    service._persist()
    restored=make_service(workspace).run_detail(job['id'])
    assert restored['status']=='complete' and restored['exportStatus']=='failed'
    assert 'interrupted' in restored['exportError']


def test_api_failure_remains_failed(workspace):
    (workspace/'main.py').write_text("import sys;print('402 Insufficient balance',flush=True);sys.exit(1)")
    service = make_service(workspace)
    d=service.add_dataset(str(workspace/'demo'/'data'))
    job=service.start_run({'featureNumber':5,'datasetId':d['id'],'question':'Measure','mode':'ultra','route':'code'})
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and service.run_detail(job['id'])['status'] in ('starting','running'):
        time.sleep(.05)
    assert service.run_detail(job['id'])['status']=='failed'


def test_artifacts_block_paths_and_filter_feature_values(workspace):
    results=workspace/'saved-run'; results.mkdir()
    (results/'features.csv').write_text('sample_id,area,intensity\na,1,9\nb,2,8\n')
    (results/'.env').write_text('SECRET=value')
    (results/'escape.txt').symlink_to(workspace/'.env')
    service=make_service(workspace)
    job=service.add_existing_run(str(results))
    evidence=service.feature_evidence(job['id'],'area')
    assert evidence['measurements']==[{'sample_id':'a','value':'1'},{'sample_id':'b','value':'2'}]
    for path in ('../.env','.env','escape.txt'):
        with pytest.raises(ValueError): service.artifact_path(job['id'],path)


@pytest.mark.parametrize('values, count, missing', [
    (['0', '0', '0'], 3, 0), (['-2', '0', '2', '', 'NaN', 'inf', 'bad'], 3, 4),
    (['7'], 1, 0), (['', 'NaN'], 0, 2), ([str(i) for i in range(10005)], 10005, 0),
])
def test_feature_distribution_counts_all_finite_measurements(workspace, values, count, missing):
    results=workspace/'distribution';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\n'+''.join(f's{i},{v}\n' for i,v in enumerate(values)))
    service=service_type()(workspace)  # Viewing results does not require API credentials.
    job=service.add_existing_run(str(results))
    histogram=service.feature_distribution(job['id'],'area')
    assert histogram['count']==count and histogram['missing']==missing
    assert sum(b['count'] for b in histogram['bins'])==count
    assert len(histogram['bins'])<=30
    assert all(b['high']>b['low'] for b in histogram['bins'])
    assert 'measurements' not in histogram
    assert histogram['feature']['status']=='retained'
    if count:
        assert histogram['minimum']==min(float(v) for v in values if v not in ('', 'NaN', 'inf', 'bad'))
        assert histogram['maximum']==max(float(v) for v in values if v not in ('', 'NaN', 'inf', 'bad'))
    else:
        assert histogram['bins']==[]
    with pytest.raises(ValueError,match='Unknown feature'):
        service.feature_distribution(job['id'],'../.env')


def test_dropped_distribution_is_not_in_portable_export(workspace):
    results=workspace/'distribution';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\na,1\nb,2\n')
    (results/'feature_registry.json').write_text(json.dumps({'entries':[
        {'name':'area','current_status':'retained','latest_round':1},
        {'name':'dropped_area','current_status':'dropped','latest_round':10}]}))
    for number,value in [(2,99),(10,3)]:
        folder=results/f'round_{number}';folder.mkdir()
        (folder/'features.csv').write_text(f'sample_id,dropped_area\na,{value}\nb,{value+1}\n')
    service=make_service(workspace)
    job=service.add_existing_run(str(results))
    with pytest.raises(ValueError, match='Unknown feature'):
        service.feature_distribution(job['id'],'dropped_area')
    exported=service.export_results(job['id'])
    feature=Path(exported['directory'])/'feature'
    value=Path(exported['directory'])/'value/feature_value.csv'
    from morphagent_ui.feature_outputs import feature_catalog
    assert 'dropped_area' not in {card['name'] for card in feature_catalog(feature)}
    assert 'dropped_area' not in {row['feature_name'] for row in csv.DictReader(value.open())}


def test_visualize_imports_standalone_feature_value_csv(workspace):
    service=make_service(workspace)
    results=workspace/'saved';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\na,1\nb,2\n')
    (results/'round_1').mkdir()
    (results/'round_1/feature_plan.json').write_text(json.dumps({'features':[
        {'name':'area','method':'code','description':'Cell area'}]}))
    source=service.add_existing_run(str(results))
    exported=service.export_results(source['id'])
    value=Path(exported['directory'])/'value/feature_value.csv'
    imported=service.add_existing_run(str(value))
    detail=service.run_detail(imported['id'])
    assert detail['features'][0]['description']=='Cell area'
    assert service.feature_distribution(imported['id'],'area')['count']==2
    with pytest.raises(ValueError,match='feature_value.csv'):
        service.add_existing_run(str(results/'features.csv'))


def test_distribution_does_not_follow_external_measurement_symlink(workspace):
    results=workspace/'distribution';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\na,1\n')
    (results/'feature_registry.json').write_text(json.dumps({'entries':[
        {'name':'area','current_status':'retained','latest_round':1},
        {'name':'secret','current_status':'retained','latest_round':1}]}))
    outside=workspace/'outside';outside.mkdir()
    (outside/'features.csv').write_text('sample_id,secret\na,123\n')
    (results/'round_1').symlink_to(outside,target_is_directory=True)
    service=make_service(workspace);job=service.add_existing_run(str(results))
    assert service.feature_distribution(job['id'],'secret')['count']==0


def test_requires_explicit_session_credentials_no_env_or_free_fallback(workspace, monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'inherited-secret')
    service=service_type()(workspace, python_executable=sys.executable)
    assert not service.settings()['hasApiKey']
    assert service.settings()['baseUrl'] == ''
    assert not hasattr(service, 'use_free_api')
    d=service.add_dataset(str(workspace/'demo/data'))
    with pytest.raises(ValueError, match='Settings'):
        service.start_run({'featureNumber':5,'datasetId':d['id'],'question':'Measure'})
    old_env=(workspace/'.env').read_bytes()
    service.save_settings({'baseUrl':'https://custom.invalid/v1','apiKey':'typed-secret','model':'my-model'})
    assert service.model_environment()['LLM_API_KEY']=='typed-secret'
    assert (workspace/'.env').read_bytes()==old_env
    assert 'typed-secret' not in (service.root/'settings.json').read_text()
    assert not service_type()(workspace).settings()['hasApiKey']


def test_partial_api_settings_do_not_inherit_legacy_key(workspace):
    service=service_type()(workspace)
    service.save_settings({'baseUrl':'https://custom.invalid/v1','model':'my-model','apiKey':''})
    d=service.add_dataset(str(workspace/'demo/data'))
    check=service.preflight({'featureNumber':5,'datasetId':d['id'],'question':'Measure'})
    assert not check['ready']
    assert any('API' in i['message'] for i in check['issues'])


def test_connection_identity_change_requires_new_key(workspace):
    service=make_service(workspace)
    service.save_settings({'baseUrl':'https://different.invalid/v1','apiKey':''})
    assert not service.settings()['hasApiKey']


def test_cancellation_and_knowledge_reach_real_inputs(workspace):
    (workspace/'demo/precomputed/deep_research_summary.txt').write_text('Prepared deep research')
    (workspace/'main.py').write_text("import time;print('Step 1: Inspect',flush=True);time.sleep(30)")
    service=make_service(workspace)
    d=service.add_dataset(str(workspace/'demo/data'))
    ref=service.add_document('notes.txt',b'Focus on neurites.')
    job=service.start_run({'featureNumber':5,'datasetId':d['id'],'question':'Measure','documentIds':[ref['id']]})
    deadline=time.monotonic()+3
    while time.monotonic()<deadline and service.run_detail(job['id'])['status']=='starting': time.sleep(.02)
    manifest=json.loads((Path(job['resultsDir'])/'ui_run_manifest.json').read_text())
    assert 'Focus on neurites.' in (Path(manifest['data_root'])/'precomputed/expert_knowledge_summary.txt').read_text()
    assert not manifest['enable_deep_research']
    assert not (Path(manifest['data_root'])/'precomputed/deep_research_summary.txt').exists()
    service.remove_document(ref['id'])
    assert 'Focus on neurites.' in (Path(manifest['data_root'])/'precomputed/expert_knowledge_summary.txt').read_text()
    assert json.loads((Path(job['resultsDir'])/'knowledge_sources.json').read_text())[0]['id']==ref['id']
    service.cancel_run(job['id'])
    deadline=time.monotonic()+3
    while time.monotonic()<deadline and service.run_detail(job['id'])['status'] in ('starting','running','cancelling'):time.sleep(.02)
    assert service.run_detail(job['id'])['status']=='cancelled'


def test_requested_deep_research_ignores_the_bundled_summary(workspace):
    (workspace/'demo/precomputed/deep_research_summary.txt').write_text('Prepared deep research')
    (workspace/'main.py').write_text("import time;print('Step 1: Inspect',flush=True);time.sleep(30)")
    service=make_service(workspace)
    d=service.add_dataset(str(workspace/'demo/data'))
    ref=service.add_document('notes.txt',b'Focus on neurites.')
    job=service.start_run({'featureNumber':5,'datasetId':d['id'],'question':'Measure',
                           'documentIds':[ref['id']],'deepResearch':True})
    deadline=time.monotonic()+3
    while time.monotonic()<deadline and service.run_detail(job['id'])['status']=='starting': time.sleep(.02)
    manifest=json.loads((Path(job['resultsDir'])/'ui_run_manifest.json').read_text())
    inputs=Path(manifest['data_root'])
    assert manifest['enable_deep_research'] and manifest['enable_expert_knowledge']
    assert '--deep-research-from-question' in manifest['command']
    assert 'Focus on neurites.' in (inputs/'precomputed/expert_knowledge_summary.txt').read_text()
    assert not (inputs/'precomputed/deep_research_summary.txt').exists()
    service.cancel_run(job['id'])
    deadline=time.monotonic()+3
    while time.monotonic()<deadline and service.run_detail(job['id'])['status'] in ('starting','running','cancelling'):time.sleep(.02)


def test_http_token_origin_and_static_isolation(workspace):
    import urllib.request
    import urllib.error
    import threading
    from morphagent_ui.web_http import create_server
    site=workspace/'design-preview';site.mkdir(parents=True)
    (site/'index.html').write_text('<html><head></head><body>Workspace</body></html>')
    server=create_server(make_service(workspace),port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    open_local=urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    try:
        assert b'morphagent-token' in open_local(url).read()
        for headers in ({}, {'X-MorphAgent-Token':'wrong'}, {'X-MorphAgent-Token':server.token,'Origin':'https://evil.invalid'}):
            with pytest.raises(urllib.error.HTTPError) as error:
                open_local(urllib.request.Request(url+'/api/bootstrap',headers=headers))
            assert error.value.code==403
        request=urllib.request.Request(url+'/api/bootstrap',headers={'X-MorphAgent-Token':server.token})
        assert b'private-test-key' not in open_local(request).read()
        with pytest.raises(urllib.error.HTTPError):open_local(url+'/.env')
    finally:
        server.shutdown();server.server_close()


def test_compute_records_prompt_but_never_passes_api_to_saved_code(workspace):
    (workspace/'.env').write_text('')
    previous=workspace/'old-results';previous.mkdir()
    (previous/'features.csv').write_text('sample_id,area\na,1\n')
    (previous/'round_1').mkdir()
    f=previous/'round_1/features/area';f.mkdir(parents=True)
    (f/'extract.py').write_text('def extract(img, seg): return 1')
    (previous/'round_1/feature_plan.json').write_text(json.dumps({'features':[{'name':'area','method':'code'}]}))
    (workspace/'reuse_code.py').write_text("import pathlib,sys,os\nassert not os.environ.get('LLM_API_KEY') and not os.environ.get('VLM_API_KEY')\np=pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]);(p/'features.csv').write_text('sample_id,area\\na,2\\n')")
    service=make_service(workspace)
    source=service.add_existing_run(str(previous))
    d=service.add_dataset(str(workspace/'demo/data'))
    j=service.start_reuse({'sourceRunId':source['id'],'datasetId':d['id'],'featureNames':['area'],'question':'Compute area on my new data'})
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        detail=service.run_detail(j['id'])
        if detail['status'] not in ('starting','running') and detail.get('exportStatus')!='saving':
            break
        time.sleep(.05)
    assert service.run_detail(j['id'])['status']=='complete'
    assert service.run_detail(j['id'])['kind']=='reuse'
    assert service.run_detail(j['id'])['selectedFeatures']==['area']
    assert service.run_detail(j['id'])['name']==j['timestamp']
    assert Path(j['resultsDir']).parent.name==j['timestamp']
    assert service.run_detail(j['id'])['question']=='Compute area on my new data'
    assert json.loads((Path(j['resultsDir'])/'ui_run_manifest.json').read_text())['query']=='Compute area on my new data'


def test_compute_loads_timestamp_export_and_checks_api_before_execution(workspace):
    from morphagent_ui.feature_outputs import export_run
    results=workspace/'raw/20260916_120000_000001/results';results.mkdir(parents=True)
    (results/'features.csv').write_text('sample_id,area\na,1\n')
    script=results/'round_1/features/area/extract.py';script.parent.mkdir(parents=True)
    script.write_text('def extract(img,seg): return 1')
    (results/'round_1/feature_plan.json').write_text(json.dumps({'features':[{'name':'area','method':'code','description':'Cell area'}]}))
    service=service_type()(workspace,python_executable=sys.executable)
    info=export_run(results,service.root/'exports','20260916_120000_000001',redact=lambda s:s)
    source=service.add_existing_run(str(Path(info['directory'])/'feature'))
    assert service.run_detail(source['id'])['features'][0]['reusable']
    assert service.run_detail(source['id'])['features'][0]['description']=='Cell area'
    nested=service.add_existing_run(str(results.parent))
    assert Path(nested['resultsDir'])==results
    data=service.add_dataset(str(workspace/'demo/data'))
    payload={'sourceRunId':source['id'],'datasetId':data['id'],'featureNames':['area'],'question':'Measure cell area'}
    check=service.preflight_compute(payload)
    assert not check['ready'] and any('API' in i['message'] for i in check['issues'])
    with pytest.raises(ValueError,match='API'):
        service.start_reuse(payload)
    assert len(service.jobs)==2
    service.save_settings({'baseUrl':'https://test.invalid/v1','apiKey':'fixture-only','model':'fixture'})
    assert service.preflight_compute(payload)['ready']
    assert not service.preflight_compute({**payload,'question':''})['ready']
    assert Path(service.bootstrap()['exportsDirectory'])==service.root/'exports'
    assert Path(service.bootstrap()['exportsDirectory']).is_dir()


def test_history_removal_trashes_exports_but_preserves_external_source(workspace,system_trash):
    service=make_service(workspace)
    results=workspace/'saved';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\na,42\n')
    job=service.add_existing_run(str(results))
    archive=service.export_results(job['id'])
    result=service.remove_run(job['id'])
    assert result['removed'] and result['id']==job['id']
    assert not service.bootstrap()['runs']
    assert job['id'] not in json.loads((service.root/'history.json').read_text())
    assert (results/'features.csv').read_text()=='sample_id,area\na,42\n'
    assert not Path(archive['archive']).exists() and not Path(archive['directory']).exists()
    assert result['systemTrash'] is True
    trash=next(system_trash.iterdir())
    assert not (service.root/'trash').exists()
    assert (trash/'exports'/Path(archive['archive']).name).is_file()
    assert (trash/'exports'/Path(archive['directory']).name/'value/feature_value.csv').is_file()
    restarted=make_service(workspace)
    assert not restarted.bootstrap()['runs']
    restored=restarted.add_existing_run(str(results))
    assert restarted.run_detail(restored['id'])['features'][0]['name']=='area'


@pytest.mark.parametrize('folder',['20260916_132213_207452','04b88b3cc5464f5fb7e05f4d73901f50'])
def test_history_removal_moves_entire_owned_run_and_all_recorded_exports(workspace,folder,system_trash):
    service=make_service(workspace)
    run=service.root/'runs'/folder;results=run/'results';results.mkdir(parents=True)
    (run/'input-copy.txt').write_text('copied input')
    (results/'features.csv').write_text('sample_id,area\na,42\n')
    job=service.add_existing_run(str(results))
    exports=[service.export_results(job['id']),service.export_results(job['id'])]
    plan=service.run_removal_plan(job['id'])
    assert str(run) in plan['paths']
    assert len(plan['paths'])==5
    result=service.remove_run(job['id'])
    assert not run.exists()
    assert result['systemTrash'] is True
    trash=next(system_trash.iterdir())
    assert not (service.root/'trash').exists()
    assert not list(service.root.glob('MorphAgent-deleted-run-*'))
    assert (trash/'runs'/folder/'input-copy.txt').read_text()=='copied input'
    manifest=json.loads((trash/'manifest.json').read_text())
    assert manifest['run']['id']==job['id']
    for export in exports:
        assert not Path(export['archive']).exists()
        assert not Path(export['directory']).exists()
    assert (workspace/'demo/data/dataset/sample_1/image.png').exists()


def test_history_removal_keeps_shared_export_and_active_compute_source(workspace):
    service=make_service(workspace)
    results=service.root/'runs/20260916_132213_207452/results';results.mkdir(parents=True)
    (results/'features.csv').write_text('sample_id,area\na,42\n')
    job=service.add_existing_run(str(results));export=service.export_results(job['id'])
    imported=service.add_existing_run(str(Path(export['directory'])/'feature'))
    service.jobs[imported['id']].update(status='running',kind='reuse',sourceRunId=job['id'])
    with pytest.raises(ValueError,match='used|active|finish'):
        service.remove_run(job['id'])
    service.jobs[imported['id']]['status']='complete'
    result=service.remove_run(job['id'])
    assert result['removed'] and not results.parent.exists()
    assert Path(export['directory']).exists() and Path(export['archive']).exists()
    assert service.run_detail(imported['id'])['features']


@pytest.mark.parametrize('target',['workspace','runs','external','symlink','input'])
def test_history_removal_never_moves_broad_external_or_input_paths(workspace,target):
    service=make_service(workspace)
    outside=workspace/'external';outside.mkdir()
    (outside/'keep.txt').write_text('keep')
    runs=service.root/'runs';runs.mkdir()
    if target=='workspace':results=service.root
    elif target=='runs':results=runs
    elif target=='external':results=outside
    elif target=='symlink':
        results=runs/'20260916_132213_207452';results.symlink_to(outside,target_is_directory=True)
    else:
        results=runs/'20260916_132213_207452/results';results.mkdir(parents=True)
        service.datasets['input']={'path':str(results.parent)}
    service.jobs['fixture']={'id':'fixture','name':'fixture','status':'complete','kind':'saved','resultsDir':str(results)}
    result=service.remove_run('fixture')
    assert result['removed']
    assert results.exists() and (outside/'keep.txt').read_text()=='keep'


def test_history_removal_rolls_back_moves_if_history_write_fails(workspace,monkeypatch):
    from morphagent_ui import web_service
    service=make_service(workspace)
    results=service.root/'runs/20260916_132213_207452/results';results.mkdir(parents=True)
    (results/'features.csv').write_text('sample_id,area\na,42\n')
    job=service.add_existing_run(str(results));original=web_service.write_json
    def fail_history(path,value):
        if path.name=='history.json':raise OSError('fixture disk failure')
        return original(path,value)
    monkeypatch.setattr(web_service,'write_json',fail_history)
    with pytest.raises(OSError,match='fixture disk failure'):service.remove_run(job['id'])
    assert job['id'] in service.jobs and (results/'features.csv').exists()


def test_system_trash_failure_restores_files_and_history(workspace,monkeypatch):
    from morphagent_ui import web_service
    service=make_service(workspace)
    results=service.root/'runs/20260916_132213_207452/results';results.mkdir(parents=True)
    (results/'features.csv').write_text('sample_id,area\na,42\n')
    job=service.add_existing_run(str(results))
    export=service.export_results(job['id'])
    def fail_trash(path):raise OSError('System Trash is unavailable')
    monkeypatch.setattr(web_service,'move_to_system_trash',fail_trash,raising=False)
    with pytest.raises(OSError,match='Trash is unavailable'):service.remove_run(job['id'])
    assert job['id'] in service.jobs
    assert job['id'] in json.loads((service.root/'history.json').read_text())
    assert (results/'features.csv').exists() and Path(export['archive']).exists()
    assert not (service.root/'trash').exists()
    assert not list(service.root.glob('MorphAgent-deleted-run-*'))


@pytest.mark.parametrize('status,export_status',[('starting',None),('running',None),('cancelling',None),('complete','saving')])
def test_history_cannot_remove_active_or_saving_run(workspace,status,export_status):
    service=make_service(workspace)
    results=workspace/'saved';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\na,42\n')
    job=service.add_existing_run(str(results))
    service.jobs[job['id']].update(status=status,exportStatus=export_status)
    with pytest.raises(ValueError,match='running|saving|finish'):
        service.remove_run(job['id'])
    assert job['id'] in service.jobs and results.exists()


def test_reuse_rejects_failed_history_empty_selection_and_unknown_features(workspace):
    results=workspace/'saved';results.mkdir()
    (results/'features.csv').write_text('sample_id,area\na,1\n')
    service=make_service(workspace);job=service.add_existing_run(str(results))
    d=service.add_dataset(str(workspace/'demo/data'))
    payload={'sourceRunId':job['id'],'datasetId':d['id'],'featureNames':[]}
    with pytest.raises(ValueError): service.start_reuse(payload)
    payload['featureNames']=['not_real']
    with pytest.raises(ValueError): service.start_reuse(payload)
    service.jobs[job['id']]['status']='failed'
    payload['featureNames']=['area']
    with pytest.raises(ValueError): service.start_reuse(payload)


def test_no_finite_measurements_is_not_reported_as_success(workspace):
    (workspace/'main.py').write_text("import pathlib,sys\np=pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]);(p/'features.csv').write_text('sample_id,area\\na,nan\\n')")
    service=make_service(workspace);d=service.add_dataset(str(workspace/'demo/data'))
    j=service.start_run({'featureNumber':5,'datasetId':d['id'],'question':'Measure'})
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and service.run_detail(j['id'])['status'] in ('starting','running'):time.sleep(.05)
    assert service.run_detail(j['id'])['status']=='empty'


def test_space_preflight_blocks_before_copy_and_execution(workspace, monkeypatch):
    from types import SimpleNamespace
    service=make_service(workspace);d=service.add_dataset(str(workspace/'demo/data'))
    monkeypatch.setattr('morphagent_ui.web_service.shutil.disk_usage',lambda _:SimpleNamespace(free=100))
    check=service.preflight({'featureNumber':5,'datasetId':d['id'],'question':'Measure'})
    assert not check['ready']
    with pytest.raises(ValueError,match='disk space'):
        service.start_run({'featureNumber':5,'datasetId':d['id'],'question':'Measure'})
    assert not service.jobs


def test_user_connection_is_only_model_environment_in_child(workspace, monkeypatch):
    monkeypatch.setenv('DEEP_RESEARCH_API_KEY','old-hidden-key')
    monkeypatch.setenv('OPENAI_API_KEY','old-openai-key')
    monkeypatch.setenv('DEEP_RESEARCH_MODEL','old-model')
    service=make_service(workspace)
    env=service._child_environment()
    env.update(service.model_environment())
    assert env['DEEP_RESEARCH_API_KEY']=='private-test-key'
    assert env['DEEP_RESEARCH_MODEL']=='test-model'
    assert env['OPENAI_API_KEY']==''


def test_terminal_history_persistence_failure_is_reported(workspace, monkeypatch):
    service=make_service(workspace)
    results=workspace/'output';results.mkdir()
    service.jobs['id']={'id':'id','status':'running','kind':'discovery','resultsDir':str(results)}
    monkeypatch.setattr(service, '_persist', lambda: (_ for _ in ()).throw(OSError(28,'No space left on device')))
    service._finish_job('id', {'status':'complete','exitCode':0,'message':'Finished'})
    assert service.jobs['id']['status']=='failed'
    assert 'persist' in service.jobs['id']['message']
