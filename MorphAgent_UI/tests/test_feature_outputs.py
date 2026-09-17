"""Selection/export contracts; no external model calls."""
import csv
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest


@pytest.fixture
def saved(tmp_path):
    root=tmp_path/'source';root.mkdir()
    definitions=[{'name':n,'method':m,'description':d} for n,m,d in
                 [('area','code','Area of cell'),('intensity','code','Mean intensity'),('visual','vlm','Visual score')]]
    round_dir=root/'round_1';round_dir.mkdir()
    (round_dir/'feature_plan.json').write_text(json.dumps({'features':definitions}))
    for name, code in [('area','def extract(img, seg): return float(img.size)\n'),
                       ('intensity',"def extract(img, seg):\n    raise RuntimeError('UNSELECTED CODE EXECUTED')\n")]:
        directory=round_dir/'features'/name;directory.mkdir(parents=True)
        (directory/'extract.py').write_text(code)
    (root/'features.csv').write_text('sample_id,area,intensity,visual\na,16,2,0.7\nb,25,3,0.8\n')
    return root


def test_catalog_only_offers_individual_code_inside_source(saved, tmp_path):
    from morphagent_ui.feature_outputs import feature_catalog
    by_name={c['name']:c for c in feature_catalog(saved)}
    assert by_name['area']['reusable'] and by_name['area']['code_status']=='available'
    # A VLM feature ships no script; it is reusable because it can be rescored.
    assert by_name['visual']['reusable'] and by_name['visual']['codePath'] is None
    assert by_name['visual']['code_status']=='vlm_scored'
    script=saved/'round_1/features/area/extract.py'
    script.unlink();script.symlink_to(tmp_path/'outside.py')
    (tmp_path/'outside.py').write_text('def extract(img,seg): return 999')
    assert not next(c for c in feature_catalog(saved) if c['name']=='area')['reusable']


def test_export_three_contents_with_matching_feature_csv(saved, tmp_path):
    from morphagent_ui.feature_outputs import export_run
    info=export_run(saved,tmp_path/'exports','20260915_123456_123456',redact=lambda s:s,run_status='partial')
    root=Path(info['directory'])
    assert root.name=='20260915_123456_123456'
    values=list(csv.DictReader((root/'value/feature_value.csv').open()))
    assert len(values)==6
    assert any(row['sample_id']=='a' and row['feature_name']=='area' and row['value']=='16' and row['description']=='Area of cell' for row in values)
    descriptions=list(csv.DictReader((root/'feature/feature_descriptions.csv').open()))
    assert next(d for d in descriptions if d['name']=='area')['description']=='Area of cell'
    assert next(d for d in descriptions if d['name']=='area')['run_status']=='partial'
    assert (root/'feature/area/code/extract.py').read_text()==(saved/'round_1/features/area/extract.py').read_text()
    assert not list(root.rglob('values.csv'))
    assert not list((root/'feature/visual/code').glob('*.py'))
    assert next(d for d in descriptions if d['name']=='visual')['code_status']=='vlm_scored'
    with zipfile.ZipFile(info['archive']) as archive:
        assert root.name+'/feature/area/code/extract.py' in archive.namelist()
        assert root.name+'/value/feature_value.csv' in archive.namelist()
        assert not any('runner.py' in name or '.env' in name for name in archive.namelist())
    # Repeated save is a new snapshot; don't overwrite a user-edited export.
    (root/'feature/area/code/extract.py').write_text('user edit')
    again=export_run(saved,tmp_path/'exports',root.name,redact=lambda s:s)
    assert Path(again['directory'])!=root
    assert (root/'feature/area/code/extract.py').read_text()=='user edit'


def test_selected_reuse_does_not_execute_unselected_code(saved, tmp_path):
    import numpy as np
    import tifffile
    from tools.selected_reuse import run_selected_reuse
    dataset=tmp_path/'data/dataset/s1';dataset.mkdir(parents=True)
    tifffile.imwrite(dataset/'image.tif',np.ones((4,4),dtype=np.uint8))
    output=tmp_path/'result'
    result=run_selected_reuse(saved,dataset.parent.parent,output,['area'])
    assert result['complete']
    assert list(csv.DictReader((output/'features.csv').open()))==[{'sample_id':'s1','area':'16.0'}]
    assert not (output/'round_1/features/intensity').exists()
    assert json.loads((output/'reuse_manifest.json').read_text())['selected_features']==['area']
    assert not json.loads((output/'reuse_manifest.json').read_text())['vlm_calls']
    with pytest.raises(ValueError):run_selected_reuse(saved,dataset.parent.parent,tmp_path/'bad',['unknown'])


@pytest.fixture
def offline_vlm_paths(monkeypatch):
    """Path selection asks the LLM which images a feature needs; keep tests offline."""
    import utils_helpers
    monkeypatch.setattr(utils_helpers,'select_appropriate_data_source',
                        lambda sample_dir,feature,description=None:[str(sample_dir/'image.tif')])


def test_selected_reuse_rescores_vlm_features_alongside_code(saved, tmp_path, monkeypatch, offline_vlm_paths):
    import numpy as np
    import tifffile
    import nodes.execution
    from tools.selected_reuse import run_selected_reuse
    dataset=tmp_path/'data/dataset/s1';dataset.mkdir(parents=True)
    tifffile.imwrite(dataset/'image.tif',np.ones((4,4),dtype=np.uint8))
    seen={}
    def fake_batch(features,image_paths,state,segmentation_mask=None,log_file=None,gpu_id=None):
        seen.update(names=[f['name'] for f in features],question=state['user_query'],
                    images=list(image_paths),sample=state['sample_id'])
        return {f['name']:71.5 for f in features}
    monkeypatch.setattr(nodes.execution,'_execute_vlm_features_batch',fake_batch)
    output=tmp_path/'mixed'
    result=run_selected_reuse(saved,dataset.parent.parent,output,['area','visual'],question='Measure tau')
    assert result['complete'] and result['vlm_calls'] and not result['llm_calls']
    # One batched call covers every selected VLM feature, exactly like a discovery run.
    assert seen['names']==['visual'] and seen['question']=='Measure tau' and seen['sample']=='s1'
    assert list(csv.DictReader((output/'features.csv').open()))==[
        {'sample_id':'s1','area':'16.0','visual':'71.5'}]
    plan=json.loads((output/'round_1/feature_plan.json').read_text())['features']
    assert {f['name']:f['method'] for f in plan}=={'area':'code','visual':'vlm'}
    registry={e['name']:e for e in json.loads((output/'feature_registry.json').read_text())['entries']}
    assert registry['visual']['method']=='vlm' and registry['visual']['current_status']=='retained'
    # No script is fabricated for a feature that never had one.
    assert not (output/'round_1/features/visual').exists()


def test_selected_reuse_records_a_failed_vlm_score(saved, tmp_path, monkeypatch, offline_vlm_paths):
    import numpy as np
    import tifffile
    import nodes.execution
    from tools.selected_reuse import run_selected_reuse
    dataset=tmp_path/'data/dataset/s1';dataset.mkdir(parents=True)
    tifffile.imwrite(dataset/'image.tif',np.ones((4,4),dtype=np.uint8))
    monkeypatch.setattr(nodes.execution,'_execute_vlm_features_batch',
                        lambda *a,**k:(_ for _ in ()).throw(RuntimeError('endpoint refused')))
    monkeypatch.setattr('tools.selected_reuse.VLM_ATTEMPTS',1)
    result=run_selected_reuse(saved,dataset.parent.parent,tmp_path/'vlm-down',['visual'],question='Measure')
    assert not result['complete']
    assert 'endpoint refused' in result['errors']['visual']['s1']


def test_selected_reuse_failure_is_not_complete(saved, tmp_path):
    import numpy as np
    import tifffile
    from tools.selected_reuse import run_selected_reuse
    dataset=tmp_path/'data/dataset/s1';dataset.mkdir(parents=True)
    tifffile.imwrite(dataset/'image.tif',np.ones((4,4),dtype=np.uint8))
    result=run_selected_reuse(saved,dataset.parent.parent,tmp_path/'partial',['area','intensity'])
    assert not result['complete']
    assert result['errors']


def test_portable_export_can_compute_without_original_run(saved, tmp_path):
    import numpy as np
    import tifffile
    from morphagent_ui.feature_outputs import export_run, feature_catalog
    from tools.selected_reuse import run_selected_reuse
    info=export_run(saved,tmp_path/'exports','20260916_130000_000001',redact=lambda s:s)
    bundle=Path(info['directory'])/'feature'
    cards={c['name']:c for c in feature_catalog(bundle)}
    assert set(cards)=={'area','intensity','visual'}
    assert cards['area']['codePath']=='area/code/extract.py'
    assert cards['area']['description']=='Area of cell'
    assert cards['area']['reusable'] and cards['visual']['reusable']
    assert cards['visual']['codePath'] is None
    # A moved bundle is self-contained; historical source paths are not required.
    saved.rename(tmp_path/'old-source-moved')
    sample=tmp_path/'new-data/dataset/new_sample';sample.mkdir(parents=True)
    tifffile.imwrite(sample/'image.tif',np.ones((3,3),dtype=np.uint8))
    result=run_selected_reuse(bundle,sample.parent.parent,tmp_path/'computed',['area'])
    assert result['complete']
    assert list(csv.DictReader((tmp_path/'computed/features.csv').open()))==[{'sample_id':'new_sample','area':'9.0'}]


def test_portable_bundle_does_not_follow_unsafe_code_folder(saved, tmp_path):
    from morphagent_ui.feature_outputs import export_run, feature_catalog
    bundle=Path(export_run(saved,tmp_path/'exports','20260916_130000_000002',redact=lambda s:s)['directory'])/'feature'
    descriptions=bundle/'feature_descriptions.csv'
    rows=list(csv.DictReader(descriptions.open()))
    rows[0]['folder']='../../outside'
    with descriptions.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    outside=tmp_path/'outside/code';outside.mkdir(parents=True)
    (outside/'extract.py').write_text('def extract(img,seg): return 999')
    cards={c['name']:c for c in feature_catalog(bundle)}
    assert 'area' in cards and not cards['area']['reusable']


def test_new_export_has_separate_feature_and_value_inputs(saved, tmp_path):
    from morphagent_ui.feature_outputs import export_run, feature_catalog, feature_distribution
    info=export_run(saved,tmp_path/'exports','20260917_120000_000001',redact=lambda s:s)
    run=Path(info['directory'])
    feature=run/'feature'
    value=run/'value/feature_value.csv'
    assert (feature/'feature_descriptions.csv').is_file()
    assert (feature/'area/code/extract.py').is_file()
    assert not list(feature.rglob('values.csv'))
    rows=list(csv.DictReader(value.open()))
    assert {'sample_id','feature_name','value','description','method'} <= set(rows[0])
    assert any(r['sample_id']=='a' and r['feature_name']=='area' and r['value']=='16' and r['description']=='Area of cell' and r['method']=='code' for r in rows)
    assert {c['name'] for c in feature_catalog(feature)}=={'area','intensity','visual'}
    assert {c['name'] for c in feature_catalog(value)}=={'area','intensity','visual'}
    assert feature_distribution(value,'area')['count']==2


def test_new_export_excludes_dropped_features(saved, tmp_path):
    from morphagent_ui.feature_outputs import export_run, feature_catalog
    registry={'entries':[{'name':name,'actual_column_name':name,'method':method,
                          'description':description,'current_status':status,'latest_round':1}
                         for name,method,description,status in
                         [('area','code','Area','retained'),('intensity','code','Intensity','dropped'),('visual','vlm','Visual','retained')]]}
    (saved/'feature_registry.json').write_text(json.dumps(registry))
    run=Path(export_run(saved,tmp_path/'exports','20260917_120000_000002',redact=lambda s:s)['directory'])
    assert {c['name'] for c in feature_catalog(run/'feature')}=={'area','visual'}
    assert {r['feature_name'] for r in csv.DictReader((run/'value/feature_value.csv').open())}=={'area','visual'}
    assert not (run/'feature/intensity').exists()
