"""Run separately from legacy PyQt5 tests: python -m unittest discover -s tests -p test_desktop_window.py."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


class DesktopWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not importlib.util.find_spec('PySide6'):
            raise unittest.SkipTest('Install requirements-desktop.txt for real Qt tests')
        if 'PyQt5.QtCore' in sys.modules:
            raise unittest.SkipTest('Run Qt6 desktop tests separately from PyQt5 tests')

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('morphagent_ui.desktop_window'), 'Desktop Qt window is missing')
        from morphagent_ui.desktop_window import create_application, WorkspaceWindow
        from morphagent_ui.desktop_runtime import DesktopRuntime
        self.app = create_application(['MorphAgent desktop test'])
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name).resolve()
        site = self.repo/'design-preview'
        site.symlink_to(Path(__file__).resolve().parents[1]/'design-preview', target_is_directory=True)
        self.runtime = DesktopRuntime(self.repo).__enter__()
        self.window = WorkspaceWindow(self.runtime)
        self.window.show()
        self.wait_js("document.title === 'MorphAgent · Workspace' && document.querySelector('.sidebar [data-page=data]') !== null")

    def tearDown(self):
        if not hasattr(self, 'window'):
            return
        self.window.close()
        self.wait(lambda:self.window.closed, timeout=15)
        self.window.deleteLater()
        from PySide6.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.runtime.stop()
        self.temp.cleanup()

    def wait(self, condition, timeout=10):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.app.processEvents()
            if condition():
                return
            time.sleep(.02)
        self.fail('Timed out waiting for Qt/WebEngine')

    def js(self, script):
        values=[]
        self.window.page.runJavaScript(script, values.append)
        self.wait(lambda:bool(values))
        return values[0]

    def wait_js(self, script):
        self.wait(lambda:self.js(script) is True, timeout=15)

    def test_real_page_transient_profile_and_empty_api(self):
        self.assertTrue(self.window.profile.isOffTheRecord())
        self.assertNotIn('PyQt5.QtCore',sys.modules)
        if destination := os.environ.get('MORPHAGENT_QA_SCREENSHOT'):
            self.window.showMaximized()
            self.app.processEvents()
            self.assertTrue(self.window.grab().save(destination))
        self.js("showSettings('api')")
        self.wait_js("document.querySelector('[data-config=baseUrl]') !== null")
        self.assertEqual(self.js("state.config.baseUrl"),'')
        self.assertEqual(self.js("state.config.apiKey"),'')
        self.assertFalse(self.runtime.service.settings()['hasApiKey'])
        self.assertEqual(self.window.view.contextMenuPolicy().name, 'NoContextMenu')

    def test_help_page_uses_session_api_and_renders_grounded_reply(self):
        self.runtime.service.save_settings({'baseUrl':'https://example.invalid/v1',
                                            'apiKey':'fixture-only-help-key','model':'fixture'})
        self.js("Object.assign(state.config,{baseUrl:'https://example.invalid/v1',model:'fixture',hasApiKey:true});document.querySelector('[data-page=help]').click()")
        self.wait_js("document.querySelector('.help-page #help-question') !== null")
        self.assertIn('I’m MorphAgent', self.js("document.querySelector('.help-page').innerText"))
        self.js("state.help.draft='What is the contribution?';render()")
        with patch('morphagent_ui.web_service.ReviewerChatClient') as client:
            client.return_value.ask.return_value='**Grounded** contribution [Manuscript].'
            self.js("document.querySelector('[data-action=help-send]').click()")
            self.wait_js("document.querySelector('.help-markdown strong')?.textContent === 'Grounded'")
            self.assertTrue(client.return_value.ask.called)
        self.assertNotIn('fixture-only-help-key',self.js('document.body.innerText'))
        self.js("document.querySelector('[data-action=help-clear]').click()")
        self.wait_js("document.querySelector('.help-page.empty') !== null")

    def test_visualize_native_upload_status_histogram_and_empty_measurements(self):
        import math
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        self.window.resize(1400,900)
        results=self.runtime.service.root/'exports'/'20260916_143000_000000'/'value'
        results.mkdir(parents=True)
        values=results/'feature_value.csv'
        values.write_text('sample_id,feature_name,value,description,method\n'+''.join(
            f's{i},neurite_skeleton_length_density,{round(2+math.sin(i)*.5+math.sin(i*.1)*.2,4)},Neurite density,code\n'
            f's{i},tau_aggregate_area_fraction,{round((i%14)/20,3)},Tau fraction,vlm\n'
            for i in range(180)))
        self.js("state.question='Unsaved question';state.featureNumber='12';document.querySelector('[data-page=visualize]').click()")
        self.wait_js("document.querySelector('.visualize-empty') !== null")
        with patch('morphagent_ui.desktop_window.QFileDialog.getOpenFileName',return_value=(str(values),'CSV files (*.csv)')) as picker:
            self.js("document.querySelector('[data-action=visualize-upload]').click()")
            self.wait_js("document.querySelector('.feature-histogram') !== null")
            self.assertEqual(picker.call_args.args[2],str(self.runtime.service.root/'exports'))
        self.assertFalse(self.runtime.service.settings()['hasApiKey'])
        self.assertTrue(self.js("document.querySelector('.sidebar [data-action=live-example], .sidebar .history-entry')===null"))
        self.assertNotIn('Imported · unverified',self.js("document.querySelector('.sidebar').textContent"))
        self.assertEqual(self.js("document.querySelectorAll('.distribution-selector .viz-feature').length"),2)
        self.assertTrue(self.js("document.querySelector('.distribution-selector').textContent.includes('Code') && document.querySelector('.distribution-selector').textContent.includes('VLM')"))
        self.assertTrue(self.js("document.querySelector('.visualize-page table, .visualize-page details, .visualize-page pre, .visualize-page img') === null"))
        self.assertTrue(self.js("document.documentElement.scrollWidth <= window.innerWidth"))
        self.assertTrue(self.js("document.querySelector('[data-action=visualize-filter][data-value=all]').getAttribute('aria-pressed')==='true'"))
        self.js("document.querySelector('[data-action=visualize-filter][data-value=code]').click()")
        self.wait_js("document.querySelectorAll('.distribution-selector .viz-feature').length === 1")
        self.js("document.querySelector('[data-action=visualize-filter][data-value=vlm]').click()")
        self.wait_js("document.querySelectorAll('.distribution-selector .viz-feature').length === 1 && document.querySelector('.feature-histogram') !== null")
        self.assertEqual(self.js("document.querySelector('.distribution-heading h2').textContent"),'tau_aggregate_area_fraction')
        if destination:=os.environ.get('MORPHAGENT_VIZ_SCREENSHOT_DIR'):
            QTest.qWait(200)  # Allow WebEngine's compositor to paint the finished chart.
            self.assertTrue(self.window.grab().save(str(Path(destination)/'visualize-filter-vlm.png')))
        self.js("document.querySelector('[data-action=visualize-filter][data-value=all]').focus()")
        QTest.keyClick(self.window.view.focusProxy(),Qt.Key.Key_Space)
        self.wait_js("document.querySelectorAll('.distribution-selector .viz-feature').length === 2")
        self.assertTrue(self.js("document.querySelector('[data-action=visualize-filter][data-value=all]').getAttribute('aria-pressed')==='true'"))
        self.js("document.querySelector('.distribution-selector .viz-feature').click()")
        self.wait_js("document.querySelector('.feature-histogram title').textContent.includes('neurite_skeleton_length_density')")
        if destination:=os.environ.get('MORPHAGENT_VIZ_SCREENSHOT_DIR'):
            self.wait_js("getComputedStyle(document.querySelector('.visualize-page')).opacity === '1'")
            self.assertTrue(self.window.grab().save(str(Path(destination)/'visualize-distribution.png')))
        self.js("[...document.querySelectorAll('.viz-feature')].find(b=>b.textContent.includes('tau_aggregate_area_fraction')).click()")
        self.wait_js("document.querySelector('.feature-histogram') !== null")
        with patch('morphagent_ui.desktop_window.QFileDialog.getOpenFileName',return_value=('','')):
            self.js("document.querySelector('[data-action=visualize-upload]').click()")
            self.assertTrue(self.js("document.querySelector('.feature-histogram') !== null"))
        with patch('morphagent_ui.desktop_window.QFileDialog.getOpenFileName',return_value=(str(self.repo/'not_feature.csv'),'')):
            self.js("document.querySelector('[data-action=visualize-upload]').click()")
            self.wait_js("state.runDialog !== null")
            self.assertIn('Unable to load run',self.js('document.body.textContent'))
        self.js("state.runDialog=null;render()")
        self.window.resize(850,750)
        self.wait_js("window.innerWidth < 900")
        self.assertTrue(self.js("document.documentElement.scrollWidth <= window.innerWidth"))
        self.assertTrue(self.js("[...document.querySelectorAll('.distribution-filter')].every(b=>b.getBoundingClientRect().right <= window.innerWidth)"))
        self.js("document.querySelector('[data-page=data]').click()")
        self.wait_js("document.querySelector('#question') !== null")
        self.assertEqual(self.js('state.question'),'Unsaved question')
        self.assertEqual(self.js('state.featureNumber'),'12')
        self.js("document.querySelector('[data-page=visualize]').click()")
        self.wait_js("document.querySelector('.visualize-empty') !== null")
        if destination:=os.environ.get('MORPHAGENT_VIZ_SCREENSHOT_DIR'):
            self.assertTrue(self.window.grab().save(str(Path(destination)/'visualize-upload.png')))

    def test_compact_pages_keep_controls_without_helper_chrome(self):
        self.window.showMaximized()
        for page in ['data', 'compute', 'settings']:
            self.js(f"navigate('{page}')")
            self.wait_js("getComputedStyle(document.querySelector('.content > section')).opacity === '1'")
            self.assertTrue(self.js("document.querySelector('.topbar, .sidebar-footnote, .composer-hint, .home-bottom') === null"))
            self.assertTrue(self.js("getComputedStyle(document.querySelector('.mobile-menu')).display === 'none'"))
            self.assertEqual(self.js("document.querySelector('.content').getBoundingClientRect().top"), 0)
            if page == 'settings':
                for tab in ['api', 'analysis']:
                    self.js(f"showSettings('{tab}')")
                    self.wait_js("getComputedStyle(document.querySelector('.settings-page')).opacity === '1'")
                    self.assertTrue(self.js("document.querySelector('.settings-footer small, .setting-note, .setting-section > p') === null"))
                    self.assertEqual(self.js("getComputedStyle(document.querySelector('.settings-body')).minHeight"), '0px')
                    self.assertTrue(self.js("document.querySelector('[data-action=save-settings]') !== null"))
                    if folder := os.environ.get('MORPHAGENT_COMPACT_SCREENSHOT_DIR'):
                        self.app.processEvents()
                        self.assertTrue(self.window.grab().save(str(Path(folder)/f'compact-settings-{tab}.png')))
            elif folder := os.environ.get('MORPHAGENT_COMPACT_SCREENSHOT_DIR'):
                self.app.processEvents()
                self.assertTrue(self.window.grab().save(str(Path(folder)/f'compact-{page}.png')))
        self.window.view.setZoomFactor(3)
        self.wait_js("window.innerWidth <= 620 && getComputedStyle(document.querySelector('.mobile-menu')).display !== 'none'")
        self.js("document.querySelector('.mobile-menu').click()")
        self.wait_js("document.querySelector('.sidebar.open') !== null")
        self.js("document.querySelector('[data-page=data]').click()")
        self.wait_js("document.querySelector('#question') !== null && state.sidebarOpen === false")

    def test_native_dataset_picker_and_cancel(self):
        from PySide6.QtWebEngineCore import QWebEnginePage
        mode=QWebEnginePage.FileSelectionMode.FileSelectUploadFolder
        with patch('morphagent_ui.desktop_window.QFileDialog.getExistingDirectory', return_value=str(self.repo)):
            self.assertEqual(self.window.page.chooseFiles(mode,[],[]),[str(self.repo)])
        with patch('morphagent_ui.desktop_window.QFileDialog.getExistingDirectory', return_value=''):
            self.assertEqual(self.window.page.chooseFiles(mode,[],[]),[])

    def test_applied_api_keys_stay_masked_and_are_not_resubmitted(self):
        from PySide6.QtTest import QTest
        self.js("Object.assign(state.config,{baseUrl:'https://test.invalid/v1',apiKey:'fixture-primary',model:'fixture',sameConnection:false,vlmBaseUrl:'https://vision.invalid/v1',vlmApiKey:'fixture-vision',vlmModel:'vision'});showSettings('api');document.querySelector('[data-action=save-settings]').click()")
        self.wait_js("document.querySelector('[data-config=apiKey]')?.readOnly === true")
        for key in ('apiKey','vlmApiKey'):
            self.assertEqual(self.js(f"document.querySelector('[data-config={key}]').value"),'********')
            self.assertEqual(self.js(f"state.config.{key}"),'')
        self.assertNotIn('fixture-primary',self.js("document.getElementById('app').innerHTML"))
        if destination := os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            QTest.qWait(250)
            self.window.grab().save(str(Path(destination)/'configured-key-mask.png'))
        self.js("document.querySelector('[data-action=save-settings]').click()")
        self.wait_js("state.savingSettings === false && document.querySelector('[data-config=apiKey]')?.readOnly === true")
        self.assertEqual(self.runtime.service.model_environment()['LLM_API_KEY'],'fixture-primary')
        self.assertEqual(self.runtime.service.model_environment()['VLM_API_KEY'],'fixture-vision')
        self.js("document.querySelector('[data-action=edit-api-key][data-key=apiKey]').click()")
        self.assertFalse(self.js("document.querySelector('[data-config=apiKey]').readOnly"))
        self.assertEqual(self.js("document.querySelector('[data-config=apiKey]').value"),'')

    def test_required_count_and_api_use_visible_modals(self):
        from PySide6.QtTest import QTest
        self.assertEqual(self.js("document.getElementById('feature-number').value"),'')
        self.js("document.querySelector('[data-action=prepare-run]').click()")
        self.wait_js("document.querySelector('#run-dialog')?.open === true")
        self.assertIn('Feature number required',self.js("document.querySelector('#run-dialog').innerText"))
        self.assertEqual(self.js("document.querySelector('#run-dialog').getAttribute('role')"),'alertdialog')
        self.assertFalse(self.runtime.service.jobs)
        self.js("document.querySelector('[data-action=resolve-run-issue]').click()")
        self.assertEqual(self.js('document.activeElement.id'),'feature-number')
        self.js("const count=document.getElementById('feature-number');count.value='12';count.dispatchEvent(new Event('input',{bubbles:true}));state.question='Measure Tau aggregation';state.dataset={id:'fixture',name:'Tau dataset',summary:{sample_count:5,primary_image_count:5,mask_count:5}};render()")
        self.assertEqual(self.js('state.featureNumber'),'12')
        if destination:=os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            QTest.qWait(250)
            self.window.grab().save(str(Path(destination)/'feature-number-home.png'))
        self.js("document.querySelector('[data-action=prepare-run]').click()")
        self.wait_js("document.querySelector('#run-dialog')?.open === true")
        self.assertIn('API key required',self.js("document.querySelector('#run-dialog').innerText"))
        if destination:=os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            QTest.qWait(250)
            self.window.grab().save(str(Path(destination)/'api-required-dialog.png'))
        self.window.resize(800,600)
        QTest.qWait(100)
        self.assertTrue(self.js("document.documentElement.scrollWidth <= window.innerWidth && document.querySelector('#run-dialog').getBoundingClientRect().right <= window.innerWidth"))
        self.assertFalse(self.runtime.service.jobs)
        self.js("document.querySelector('[data-action=resolve-run-issue]').click()")
        self.wait_js("state.page === 'settings' && !document.querySelector('#run-dialog')")
        self.assertTrue(self.js("document.querySelector('[data-config=apiKey]') !== null"))

    def test_knowledge_attachment_dialog_pdf_preview_download_and_remove(self):
        import base64
        import io
        from PySide6.QtTest import QTest
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer=PdfWriter()
        page=writer.add_blank_page(width=400,height=300)
        font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream=DecodedStreamObject();stream.set_data(b'BT /F1 12 Tf 20 250 Td (Focus on Tau aggregation.) Tj ET')
        page[NameObject('/Contents')]=writer._add_object(stream)
        data=io.BytesIO();writer.write(data)
        pdf=self.repo/'Tau research.pdf';pdf.write_bytes(data.getvalue())
        note=self.repo/'Expert notes.txt';note.write_text('Inspect neuronal morphology.\n<script>not executable</script>')
        self.js("document.querySelector('[data-action=manage-knowledge]').click()")
        self.wait_js("document.querySelector('#knowledge-dialog')?.open === true")
        # Feed the real HTML file input; native OS chooser dialogs are not mocked
        # because Qt captures virtual method overrides when the page is created.
        files=[{'name':p.name,'base64':base64.b64encode(p.read_bytes()).decode()} for p in (pdf,note)]
        self.js("(()=>{const data="+json.dumps(files)+";const transfer=new DataTransfer();for(const f of data){transfer.items.add(new File([Uint8Array.from(atob(f.base64),c=>c.charCodeAt(0))],f.name));}const input=document.getElementById('knowledge-input');input.files=transfer.files;input.dispatchEvent(new Event('change',{bubbles:true}));})()")
        self.wait_js('state.docs.length === 2 && !state.knowledge.uploading')
        self.assertEqual(len(self.runtime.service.documents),2)
        self.js("document.querySelector('[data-action=close-knowledge]').click()")
        self.wait_js("!document.querySelector('#knowledge-dialog') && document.querySelector('[data-action=manage-knowledge]').innerText.includes('Knowledge attached')")
        self.assertEqual(self.js('document.activeElement.dataset.action'),'manage-knowledge')
        if destination:=os.environ.get('MORPHAGENT_KNOWLEDGE_SCREENSHOT_DIR'):
            QTest.qWait(250)
            self.window.grab().save(str(Path(destination)/'knowledge-home.png'))
        self.js("document.querySelector('[data-action=manage-knowledge]').click()")
        self.wait_js("document.querySelector('#knowledge-dialog')?.open === true")
        self.assertIn('Tau research.pdf',self.js("document.querySelector('#knowledge-dialog').innerText"))
        if destination:=os.environ.get('MORPHAGENT_KNOWLEDGE_SCREENSHOT_DIR'):
            QTest.qWait(250)
            self.window.grab().save(str(Path(destination)/'knowledge-dialog.png'))
        self.js("document.querySelector('[data-action=preview-doc]').click()")
        self.wait_js("document.querySelector('.knowledge-text')?.textContent.includes('Focus on Tau aggregation.')")
        saved=self.repo/'downloaded.pdf'
        with patch('morphagent_ui.desktop_window.QFileDialog.getSaveFileName',return_value=(str(saved),'')):
            self.js("document.querySelector('[data-action=download-doc]').click()")
            self.wait(lambda:saved.is_file() and saved.stat().st_size==len(data.getvalue()))
        self.assertEqual(saved.read_bytes(),data.getvalue())
        self.js("document.querySelector('[data-action=knowledge-list]').click()")
        self.js("document.querySelector('[data-action=remove-doc]').click()")
        self.wait_js('state.docs.length === 1')
        self.js("document.querySelector('[data-action=preview-doc]').click()")
        self.wait_js("document.querySelector('.knowledge-text')?.textContent.includes('<script>not executable</script>')")
        self.assertTrue(self.js("document.querySelector('.knowledge-text script') === null"))
        self.js("document.querySelector('[data-action=knowledge-list]').click()")
        self.js("document.querySelector('[data-action=remove-doc]').click()")
        self.wait_js("state.docs.length === 0 && document.querySelector('#knowledge-dialog').innerText.includes('No knowledge attached')")
        self.assertFalse(self.runtime.service.documents)
        self.assertEqual(len(list((self.repo/'.web_workspace/references').glob('*/*.*'))),4)
        self.js("document.querySelector('[data-action=close-knowledge]').click()")
        self.wait_js("document.querySelector('[data-action=manage-knowledge]').innerText.includes('Upload knowledge')")
        self.window.resize(800,600)
        self.app.processEvents()
        self.assertTrue(self.js('document.documentElement.scrollWidth <= window.innerWidth'))
        self.js("showSettings('api')")
        self.assertTrue(self.js("document.querySelector('[data-tab=knowledge]') === null"))

    def test_folder_upload_through_real_webengine(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        sample=self.repo/'incoming/dataset/sample_1'
        sample.mkdir(parents=True)
        (sample/'image.tif').write_bytes(b'fixture-only')
        with patch('morphagent_ui.desktop_window.QFileDialog.getExistingDirectory',return_value=str(self.repo/'incoming')) as picker:
            rect=json.loads(self.js("JSON.stringify((()=>{const r=document.querySelector('[data-action=pick-data]').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})())"))
            QTest.mouseClick(self.window.view.focusProxy() or self.window.view,Qt.MouseButton.LeftButton,pos=QPoint(int(rect[0]),int(rect[1])))
            self.wait(lambda:picker.called)
            self.wait_js('state.dataset?.summary?.sample_count === 1')
        path=self.js('state.dataset.path')
        self.assertTrue(Path(path).is_relative_to(self.repo/'.web_workspace/imports'))
        self.assertTrue(list(Path(path).rglob('image.tif')))

    def test_data_submit_and_export_in_desktop_without_external_api(self):
        # Fixture CLI proves the UI plumbing only; it is not a scientific run.
        sample=self.repo/'demo/data/dataset/one'
        sample.mkdir(parents=True)
        (sample/'image.tif').write_bytes(b'fixture')
        (self.repo/'main.py').write_text('''import json,sys,time
from pathlib import Path
r=Path(sys.argv[sys.argv.index('--results-dir')+1])
print('Step 4: Quantify',flush=True)
time.sleep(3)
(r/'features.csv').write_text('sample_id,area\\none,42\\n')
p=r/'round_1'; p.mkdir()
(p/'feature_plan.json').write_text(json.dumps({'features':[{'name':'area','description':'Test area','method':'code'}]}))
f=p/'features/area'; f.mkdir(parents=True)
(f/'extract.py').write_text('def extract(image, masks):\\n    return 42.0\\n')
print('Final feature file: '+str(r/'features.csv'),flush=True)
''')
        self.js("Object.assign(state.config,{baseUrl:'https://test.invalid/v1',model:'fixture',apiKey:'fixture-not-a-real-key',route:'code',knowledgeEnabled:false});document.querySelector('[data-action=demo]').click()")
        self.wait_js('state.dataset?.summary?.sample_count === 1')
        self.js("state.question='Fixture plumbing test'; state.featureNumber='7'; document.querySelector('[data-action=prepare-run]').click()")
        self.wait_js("document.querySelector('#run-dialog')?.open === true && state.runDialog.kind === 'confirm'")
        self.assertFalse(self.runtime.service.jobs)
        self.assertIn('Fixture plumbing test',self.js("document.querySelector('#run-dialog').innerText"))
        if destination:=os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            from PySide6.QtTest import QTest
            QTest.qWait(250)
            self.window.grab().save(str(Path(destination)/'run-confirmation.png'))
        self.js("document.querySelector('[data-action=close-run-dialog]').click()")
        self.assertFalse(self.runtime.service.jobs)
        self.assertEqual(self.js('state.featureNumber'),'7')
        self.js("document.querySelector('[data-action=prepare-run]').click()")
        self.wait_js("document.querySelector('#run-dialog')?.open === true && state.runDialog.kind === 'confirm'")
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        QTest.keyClick(self.window.view.focusProxy() or self.window.view,Qt.Key.Key_Escape)
        self.wait_js('state.runDialog === null')
        self.assertFalse(self.runtime.service.jobs)
        self.js("document.getElementById('question').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true}))")
        self.wait_js("document.querySelector('#run-dialog')?.open === true && state.runDialog.kind === 'confirm'")
        # Hold packaging briefly to verify polling continues after CLI completion.
        import threading
        export_ready=threading.Event()
        original_export=self.runtime.service.export_results
        def delayed_export(*args, **kwargs):
            export_ready.wait(timeout=10)
            return original_export(*args, **kwargs)
        patcher=patch.object(self.runtime.service,'export_results',side_effect=delayed_export)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(export_ready.set)
        self.js("document.querySelector('[data-action=confirm-run]').click()")
        self.wait_js("document.querySelector('.eta-panel')?.innerText.includes('About')")
        self.assertEqual(self.js('state.page'),'data')
        self.assertEqual(self.js("document.querySelector('.submitted-question h1').innerText"),'Fixture plumbing test')
        self.assertTrue(self.js("document.querySelector('[data-page=run]') === null && document.querySelector('#question') === null"))
        if destination := os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            QTest.qWait(250)
            self.window.grab().save(str(Path(destination)/'feature-design-running.png'))
        self.wait_js("document.querySelector('.automatic-results')?.innerText.includes('Saving results')")
        self.assertEqual(self.js('state.page'),'data')
        export_ready.set()
        self.wait(lambda:bool(self.runtime.service.jobs) and all(j['status']=='complete' and j.get('exportStatus')=='ready' for j in self.runtime.service.jobs.values()))
        self.assertEqual(len(self.runtime.service.jobs),1)
        self.assertEqual(next(iter(self.runtime.service.jobs.values()))['featureNumber'],7)
        self.wait_js("document.querySelector('.run-status')?.innerText.includes('Pipeline finished')")
        self.assertNotIn('PyQt5.QtCore',sys.modules)
        self.wait_js("document.querySelector('[data-action=download-bundle]') !== null")
        self.assertEqual(self.js('state.page'),'data')
        self.assertIn('Results saved',self.js("document.querySelector('.automatic-results').innerText"))
        self.window.resize(800,850)
        self.app.processEvents()
        self.assertTrue(self.js("document.documentElement.scrollWidth <= window.innerWidth"))
        self.window.resize(1400,900)
        if destination := os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            QTest.qWait(250)
            self.window.grab().save(str(Path(destination)/'feature-design-completed.png'))
        destination=self.repo/'fixture-results.zip'
        with patch('morphagent_ui.desktop_window.QFileDialog.getSaveFileName',return_value=(str(destination),'')):
            self.js("document.querySelector('[data-action=download-bundle]').click()")
            self.wait(lambda:destination.is_file() and destination.stat().st_size>0)
        import zipfile
        with zipfile.ZipFile(destination) as archive:
            names=archive.namelist()
            self.assertTrue(any(n.endswith('/value/feature_value.csv') for n in names))
            self.assertTrue(any(n.endswith('/feature/feature_descriptions.csv') for n in names))
            self.assertTrue(any(n.endswith('/feature/area/code/extract.py') for n in names))
        self.js("document.querySelector('[data-page=data]').click()")
        self.wait_js("document.querySelector('#question') !== null")
        self.assertEqual(self.js('state.question'), '')
        self.assertEqual(self.js('state.featureNumber'), '')
        self.assertTrue(self.js('state.dataset === null'))
        self.assertEqual(len(self.runtime.service.jobs),1)

    def test_compute_picker_confirmation_and_inline_results(self):
        from morphagent_ui.feature_outputs import export_run
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        source=self.repo/'old-results'
        script=source/'round_1/features/area/extract.py'
        script.parent.mkdir(parents=True)
        script.write_text('def extract(img,seg): return float(img.size)\n')
        (source/'round_1/feature_plan.json').write_text(json.dumps({'features':[{'name':'area','method':'code','description':'Area of cell'}]}))
        (source/'features.csv').write_text('sample_id,area\nold_sample,16\n')
        saved=export_run(source,self.runtime.service.root/'exports','20260916_130000_000001',redact=lambda s:s)
        sample=self.repo/'incoming/dataset/new_sample';sample.mkdir(parents=True)
        (sample/'image.tif').write_bytes(b'fixture-only')
        # Exercise actual desktop/HTTP/subprocess/export plumbing with no model call.
        (self.repo/'reuse_code.py').write_text('''import json,os,shutil,sys,time
from pathlib import Path
assert not os.environ.get('LLM_API_KEY') and not os.environ.get('VLM_API_KEY')
assert sys.argv[sys.argv.index('--features')+1:]==['area']
source=Path(sys.argv[sys.argv.index('--source-results')+1])
r=Path(sys.argv[sys.argv.index('--results-dir')+1])
print('[Reuse] area · new_sample',flush=True)
time.sleep(2)
p=r/'round_1';p.mkdir()
(p/'feature_plan.json').write_text(json.dumps({'features':[{'name':'area','method':'code'}]}))
f=p/'features/area';f.mkdir(parents=True)
shutil.copy2(source/'area/code/extract.py',f/'extract.py')
(r/'features.csv').write_text('sample_id,area\\nnew_sample,9\\n')
print('[Compute] Progress 1/1',flush=True)
''')
        self.js("state.question='Keep this design draft';navigate('compute')")
        self.assertTrue(self.js("document.querySelector('[data-page=save]') === null && document.querySelector('[data-page=reuse]') === null"))
        self.assertEqual(self.js("document.querySelector('.sidebar-actions').innerText"),'Design\nCompute\nVisualize\nHelp')
        self.assertTrue(self.js("document.querySelector('.topbar [data-page]') === null"))
        with patch('morphagent_ui.desktop_window.QFileDialog.getExistingDirectory',return_value=str(Path(saved['directory'])/'feature')) as picker:
            self.js("document.querySelector('[data-action=compute-source]').click()")
            self.wait_js("document.querySelector('.compute-attachments').innerText.includes('20260916_130000')")
            self.assertEqual(picker.call_args.args[2],str(self.runtime.service.root/'exports'))
        with patch('morphagent_ui.desktop_window.QFileDialog.getExistingDirectory',return_value=''):
            self.js("document.querySelector('[data-action=compute-source]').click()")
            self.assertIn('20260916_130000',self.js("document.querySelector('.compute-attachments').innerText"))
        with patch('morphagent_ui.desktop_window.QFileDialog.getExistingDirectory',return_value=str(self.repo/'incoming')) as picker:
            rect=json.loads(self.js("JSON.stringify((()=>{const r=document.querySelector('[data-action=reuse-upload]').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})())"))
            QTest.mouseClick(self.window.view.focusProxy() or self.window.view,Qt.MouseButton.LeftButton,pos=QPoint(int(rect[0]),int(rect[1])))
            self.wait(lambda:picker.called)
            self.wait_js("document.querySelector('.compute-attachments').innerText.includes('New dataset · 1 samples')")
        self.js("document.querySelector('[data-action=compute-features]').click()")
        self.assertTrue(self.js("document.querySelector('.compute-feature-picker').innerText.includes('area')"))
        self.assertTrue(self.js("document.querySelector('.compute-feature-picker input[type=checkbox]') === null"))
        self.assertTrue(self.js("document.querySelector('[data-action=reuse-all]') === null && document.querySelector('[data-action=reuse-clear]') === null"))
        # Compute replays a fixed feature set, so the page asks for nothing else.
        self.assertTrue(self.js("document.querySelector('.compute-home textarea') === null"))
        self.js("document.querySelector('[data-action=prepare-compute]').click()")
        self.wait_js("document.querySelector('#run-dialog')?.innerText.includes('API key required')")
        self.assertEqual(len(self.runtime.service.jobs),1)
        self.js("document.querySelector('[data-action=close-run-dialog]').click();Object.assign(state.config,{baseUrl:'https://test.invalid/v1',apiKey:'fixture-only-key',model:'fixture'});document.querySelector('[data-action=prepare-compute]').click()")
        self.wait_js("state.runDialog?.kind === 'confirm' && document.querySelector('#run-dialog')?.open === true")
        content=self.js("document.querySelector('#run-dialog').innerText")
        for text in ('Previous run','20260916_130000','area','1 samples','Compute'):
            self.assertIn(text,content)
        self.assertNotIn('fixture-only-key',content)
        self.js("document.querySelector('[data-action=close-run-dialog]').click()")
        self.assertEqual(len(self.runtime.service.jobs),1)
        self.window.resize(850,850);QTest.qWait(250)
        self.assertTrue(self.js('document.documentElement.scrollWidth <= window.innerWidth'))
        self.window.resize(1400,960);QTest.qWait(250)
        if destination:=os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            self.window.grab().save(str(Path(destination)/'compute-ready.png'))
        self.js("document.querySelector('[data-action=prepare-compute]').click()")
        self.wait_js("state.runDialog?.kind === 'confirm'")
        self.js("document.querySelector('[data-action=confirm-run]').click()")
        self.wait_js("document.querySelector('.eta-panel')?.innerText.includes('About')")
        self.assertEqual(self.js('state.page'),'compute')
        self.assertTrue(self.js("document.querySelector('.submitted-question h1').innerText.includes('Apply the saved MorphAgent features')"))
        self.assertEqual(self.js('state.question'),'Keep this design draft')
        self.assertTrue(self.js('state.dataset === null'))
        if destination:=os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            QTest.qWait(250);self.window.grab().save(str(Path(destination)/'compute-running.png'))
        self.wait_js("document.querySelector('[data-action=download-bundle]') !== null")
        self.assertEqual(self.js('state.page'),'compute')
        self.assertEqual(len(self.runtime.service.jobs),2)
        job=next(j for j in self.runtime.service.jobs.values() if j['kind']=='reuse')
        self.assertEqual(job['status'],'complete')
        self.assertEqual(job['completedMeasurements'],1)
        self.assertEqual(job['exportStatus'],'ready')
        if destination:=os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            QTest.qWait(250);self.window.grab().save(str(Path(destination)/'compute-completed.png'))
        output=self.repo/'compute-results.zip'
        with patch('morphagent_ui.desktop_window.QFileDialog.getSaveFileName',return_value=(str(output),'')):
            self.js("document.querySelector('[data-action=download-bundle]').click()")
            self.wait(lambda:output.is_file() and output.stat().st_size>0)
        self.js("document.querySelector('[data-page=compute]').click()")
        self.wait_js("document.querySelector('.compute-home') !== null")
        self.assertNotIn('Previous run attached', self.js("document.querySelector('.compute-home').innerText"))
        self.assertNotIn('Data attached', self.js("document.querySelector('.compute-home').innerText"))
        self.assertEqual(self.js('state.question'),'Keep this design draft')

    def test_history_view_is_separate_from_both_new_run_drafts(self):
        results=self.repo/'previous/results';results.mkdir(parents=True)
        (results/'features.csv').write_text('sample_id,area\ncell_1,42\n')
        job=self.runtime.service.add_existing_run(str(results))
        self.runtime.service.jobs[job['id']].update(kind='discovery',status='complete')
        self.window.view.reload()
        self.wait_js("document.querySelector('[data-action=load-job]') !== null")
        self.js("document.querySelector('[data-action=load-job]').click()")
        self.wait_js("document.querySelector('.submitted-question') !== null")
        self.js("document.querySelector('[data-page=data]').click()")
        self.wait_js("document.querySelector('#question') !== null")
        self.assertEqual(self.js('state.question'),'')
        self.assertTrue(self.js('state.dataset === null'))
        self.js("state.question='Unsaved design';state.featureNumber='12';state.dataset={id:'draft',name:'Draft dataset',summary:{sample_count:1,primary_image_count:1,mask_count:0}};document.querySelector('[data-page=compute]').click()")
        self.wait_js("document.querySelector('.compute-home') !== null")
        self.js("document.querySelector('[data-action=load-job]').click()")
        self.wait_js("document.querySelector('.submitted-question') !== null")
        self.assertNotIn('Unsaved design', self.js("document.querySelector('.submitted-question').innerText"))
        self.assertNotIn('Draft dataset', self.js("document.querySelector('.run-meta').innerText"))
        self.js("document.querySelector('[data-page=data]').click()")
        self.wait_js("document.getElementById('question')?.value === 'Unsaved design'")
        self.assertTrue(self.js("document.querySelector('.history-entry.active, .submitted-question') === null"))
        # Compute keeps no typed draft, so it returns to its own empty composer.
        self.js("document.querySelector('[data-page=compute]').click()")
        self.wait_js("document.querySelector('.compute-home') !== null")
        self.assertTrue(self.js("document.querySelector('.history-entry.active, .submitted-question') === null"))
        self.assertEqual(self.js('state.featureNumber'),'12')
        self.assertEqual(self.js('state.dataset.id'),'draft')
        self.assertEqual(list(self.runtime.service.jobs),[job['id']])
        self.assertTrue((results/'features.csv').exists())

    def test_history_delete_dialog_removes_owned_folders_to_recoverable_trash(self):
        from PySide6.QtTest import QTest
        system_trash=self.repo/'fake-system-trash';system_trash.mkdir()
        def trash_move(path):
            destination=system_trash/Path(path).name
            Path(path).rename(destination)
            return str(destination)
        trash_patch=patch('morphagent_ui.web_service.move_to_system_trash',side_effect=trash_move)
        trash_patch.start();self.addCleanup(trash_patch.stop)
        results=self.runtime.service.root/'runs/20260916_132213_207452/results';results.mkdir(parents=True)
        (results/'features.csv').write_text('sample_id,area\ncell_1,42\n')
        job=self.runtime.service.add_existing_run(str(results))
        self.runtime.service.jobs[job['id']].update(kind='discovery',status='complete')
        archive=self.runtime.service.export_results(job['id'])
        self.window.view.reload()
        self.wait_js("document.querySelector('[data-action=load-job]') !== null")
        self.js("document.querySelector('[data-action=load-job]').click()")
        self.wait_js("document.querySelector('.automatic-results')?.innerText.includes('Results saved')")
        self.assertTrue(self.js("document.querySelector('[data-action=live-new]') === null"))
        self.js("document.querySelector('[data-action=delete-history]').click()")
        self.wait_js("document.querySelector('#run-dialog')?.open === true")
        text=self.js("document.querySelector('#run-dialog').innerText")
        self.assertIn('Delete run?',text)
        self.assertIn('system Trash / Recycle Bin',text)
        self.assertNotIn('.web_workspace/trash',text)
        self.assertIn(str(results.parent),text)
        self.assertIn(archive['archive'],text)
        self.assertIn(job['id'],self.runtime.service.jobs)
        if destination:=os.environ.get('MORPHAGENT_RUN_SCREENSHOT_DIR'):
            QTest.qWait(250);self.window.grab().save(str(Path(destination)/'history-removal.png'))
        self.js("document.querySelector('[data-action=close-run-dialog]').click()")
        self.assertIn(job['id'],self.runtime.service.jobs)
        self.assertTrue(results.exists())
        self.js("document.querySelector('[data-action=delete-history]').click()")
        self.wait_js("document.querySelector('#run-dialog')?.open === true")
        self.js("document.querySelector('[data-action=confirm-delete-history]').click()")
        self.wait_js("document.querySelector('[data-action=load-job]') === null && state.runDialog === null")
        self.assertNotIn(job['id'],self.runtime.service.jobs)
        self.assertFalse(results.parent.exists())
        self.assertFalse(Path(archive['archive']).exists())
        trash=next(system_trash.iterdir())
        self.assertFalse((self.runtime.service.root/'trash').exists())
        self.assertTrue((trash/'runs'/results.parent.name/'results/features.csv').exists())
        self.assertTrue((trash/'exports'/Path(archive['archive']).name).exists())
        self.assertTrue(self.js("document.querySelector('[data-action=download-bundle]') === null"))
        self.assertTrue(self.js("document.getElementById('question') !== null"))
        self.window.view.reload()
        self.wait_js("document.querySelector('.history-empty') !== null")
        self.assertTrue(self.js("document.querySelector('[data-action=load-job]') === null"))

    def test_native_system_trash_moves_only_the_fixture_folder(self):
        from morphagent_ui.web_service import move_to_system_trash
        folder=self.repo/'MorphAgent-system-trash-test';folder.mkdir()
        (folder/'fixture.txt').write_text('temporary fixture only')
        location=move_to_system_trash(folder)
        destination=Path(location) if location else None
        try:
            self.assertFalse(folder.exists())
            if destination:
                self.assertEqual((destination/'fixture.txt').read_text(),'temporary fixture only')
        finally:
            # Restore only this test-owned item; never empty or enumerate the user's Trash.
            if destination and destination.exists():destination.rename(folder)

    def test_native_trash_failure_does_not_permanently_delete(self):
        from morphagent_ui.web_service import move_to_system_trash
        folder=self.repo/'cannot-trash-fixture';folder.mkdir()
        with patch('PySide6.QtCore.QFile') as qfile:
            qfile.supportsMoveToTrash.return_value=True
            qfile.return_value.moveToTrash.return_value=False
            qfile.return_value.errorString.return_value='fixture permission failure'
            with self.assertRaisesRegex(OSError,'Could not move'):
                move_to_system_trash(folder)
        self.assertTrue(folder.exists())

    def test_prompt_uses_native_folder_and_external_navigation_is_blocked(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtWebEngineCore import QWebEnginePage
        with patch('morphagent_ui.desktop_window.QFileDialog.getExistingDirectory', return_value=str(self.repo)):
            result=self.window.page.javaScriptPrompt(QUrl(self.runtime.url),'Paste the local results folder path (contains features.csv):','')
            self.assertEqual(result,(True,str(self.repo)))
        self.assertFalse(self.window.page.acceptNavigationRequest(QUrl('file:///etc/passwd'),QWebEnginePage.NavigationType.NavigationTypeOther,True))
        self.assertFalse(self.window.page.acceptNavigationRequest(QUrl('https://example.com'),QWebEnginePage.NavigationType.NavigationTypeOther,True))

    def test_real_blob_download_and_save_dialog(self):
        destination=self.repo/'downloaded.csv'
        with patch('morphagent_ui.desktop_window.QFileDialog.getSaveFileName', return_value=(str(destination),'')):
            self.js("download('values.csv','sample_id,area\\na,42\\n','text/csv')")
            self.wait(lambda:destination.is_file() and destination.read_text()=='sample_id,area\na,42\n')
        self.assertEqual(destination.read_text(),'sample_id,area\na,42\n')

    def test_active_close_cancel_keeps_window_and_service(self):
        from PySide6.QtWidgets import QMessageBox
        with patch.object(self.runtime,'has_active_runs',return_value=True), patch('morphagent_ui.desktop_window.QMessageBox.question',return_value=QMessageBox.StandardButton.Cancel):
            self.window.close()
            self.app.processEvents()
            self.assertFalse(self.window.closed)
            self.assertTrue(self.window.isVisible())
            self.assertTrue(self.runtime.thread.is_alive())

    def test_system_quit_respects_active_run_confirmation(self):
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtCore import QCoreApplication, QEvent
        with patch.object(self.runtime,'has_active_runs',return_value=True), patch('morphagent_ui.desktop_window.QMessageBox.question',return_value=QMessageBox.StandardButton.Cancel) as question:
            QCoreApplication.sendEvent(self.app,QEvent(QEvent.Type.Quit))
            self.app.processEvents()
            self.assertTrue(question.called)
            self.assertFalse(self.window.closed)
            self.assertTrue(self.runtime.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
