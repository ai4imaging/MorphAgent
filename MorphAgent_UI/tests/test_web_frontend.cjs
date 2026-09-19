const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const flush = () => new Promise(resolve => setImmediate(resolve));

async function ui({failDocumentName='',additionalCode=false,historyKind='reuse',onlyRetained=false,includeImports=false,onlyImports=false}={}) {
  const elements=new Map();
  const element=id=>{
    if(!elements.has(id)) elements.set(id,{innerHTML:'',textContent:'',dataset:{},value:'',handlers:[],
      classList:{add(){},remove(){}},querySelector(){return null;},focus(){},click(){},showModal(){this.open=true;},
      addEventListener(type,fn,capture){this.handlers.push({type,fn,capture});}});
    return elements.get(id);
  };
  const calls=[];
  const documents=[];
  const settings={baseUrl:'',model:'',apiKey:'',hasApiKey:false,vlmApiKey:'',hasVlmApiKey:false,route:'code',mode:'ultra',sameConnection:true,knowledgeEnabled:false};
  const runs=[{id:'ok',name:'20260915_120000_000000',timestamp:'20260915_120000_000000',status:'complete',startedAt:1},
    {id:'bad',name:'failed-run',status:'failed',startedAt:2}];
  if(onlyImports)runs.splice(0);
  if(includeImports||onlyImports)runs.push(
    {id:'imported',name:'imported-results',kind:'saved',status:'loaded'},
    {id:'demo-import',name:'completed_demo_run',kind:'saved',status:'loaded'},
    {id:'legacy-import',name:'legacy-loaded-run',status:'loaded'});
  const context={console,Blob,URL,setTimeout(){},clearTimeout(){},Set,Date,
    document:{getElementById:element,querySelector:s=>s.includes('morphagent-token')?{content:'test-token'}:null,addEventListener(){},createElement:()=>element('anchor')},
    window:{scrollTo(){},prompt(message,initial){calls.push({endpoint:'native-picker',message,initial});return '/workspace/exports/20260915_120000_000000';},confirm(){throw Error('Unexpected extra confirmation');}},
    fetch:async(url,options={})=>{
      const endpoint=url.replace('/api/',''),body=options.body && !(options.body instanceof Blob)?JSON.parse(options.body):{};
      calls.push({endpoint,body});
      let data={};
      if(endpoint==='bootstrap')data={settings,runs,datasets:[],documents:[...documents],exportsDirectory:'/workspace/exports'};
      if(endpoint.startsWith('documents?name=')){
        const name=decodeURIComponent(endpoint.split('=')[1]);
        if(name===failDocumentName)return {ok:false,json:async()=>({error:'No readable text found.'})};
        data={id:'doc'+documents.length,name,size:options.body.size,characters:24};documents.push(data);
      }
      if(endpoint.match(/^documents\/[^/]+$/))data={...documents.find(d=>d.id===endpoint.split('/')[1]),text:'Extracted scientific reference.'};
      if(endpoint.startsWith('documents/')&&endpoint.endsWith('/remove')){documents.splice(documents.findIndex(d=>d.id===endpoint.split('/')[1]),1);data={removed:true};}
      if(endpoint.startsWith('runs/')&&endpoint.endsWith('/removal'))data={paths:['/workspace/runs/20260915_120000_000000','/workspace/exports/20260915_120000_000000.zip'],preservedPaths:[]};
      if(endpoint.startsWith('runs/')&&endpoint.endsWith('/remove')){runs.splice(runs.findIndex(r=>r.id===endpoint.split('/')[1]),1);data={removed:true,systemTrash:true};}
      if(endpoint==='settings'){Object.assign(settings,body);data={...settings,hasApiKey:!!body.apiKey||settings.hasApiKey};}
      if(endpoint==='help/ask')data={answer:'**Grounded** answer [Manuscript]. <script>alert(1)</script>'};
      if(endpoint==='preflight')data={ready:true,issues:[],rounds:1,target:5,samples:1};
      if(endpoint==='compute/preflight')data={ready:true,issues:[],features:1,samples:2};
      if(endpoint==='runs/load')data={id:'ok'};
      if(endpoint==='datasets/demo')data={id:'target',name:'New dataset',summary:{sample_count:2,primary_image_count:2}};
      if(endpoint==='datasets')data={id:'by-path',name:'picked',path:body.path,summary:{sample_count:4,primary_image_count:4,mask_count:0}};
      if(endpoint==='imports')data={id:'up1'};
      if(endpoint==='imports/up1/finish')data={id:'uploaded',name:'copied',path:'/workspace/.web_workspace/imports/up1',summary:{sample_count:1,primary_image_count:1,mask_count:0}};
      if(endpoint==='compute'){data={id:'computed',kind:'reuse',status:'running',question:body.question};runs.push(data);}
      if(endpoint==='runs/computed')data={id:'computed',kind:'reuse',status:'running',question:'Compute cell area',name:'computed',features:[],artifacts:[],startedAt:Date.now()/1000,eta:{remainingSeconds:30},selectedFeatures:['area']};
      if(endpoint==='runs'){data={id:'new',kind:'discovery',status:'running',question:body.question};runs.push(data);}
      if(endpoint==='runs/new')data={id:'new',status:'running',startedAt:Date.now()/1000,question:'Measure',name:'timestamp',features:[],artifacts:[],eta:{remainingSeconds:300,progressPercent:20}};
      if(endpoint==='runs/ok')data={...runs[0],kind:historyKind,question:'Historical question',selectedFeatures:['area'],features:[{name:'area',description:'Cell area',method:'code',status:'retained',reusable:true},{name:'visual',description:'Visual score',method:'vlm',status:'retained',reusable:true},{name:'dropped',method:'code',status:'dropped',reusable:false}],artifacts:[],exportStatus:'ready',lastExport:{directory:'/workspace/exports/20260915_120000_000000',filename:'20260915_120000_000000.zip'}};
      if(endpoint.includes('/distribution?feature='))data={count:3,missing:0,minimum:0,maximum:2,bins:[{low:0,high:1,count:1},{low:1,high:2,count:2}]};
      if(endpoint.includes('/logs'))data={lines:[],offset:0};
      if(endpoint==='runs/ok' && additionalCode)data.features.push({name:'perimeter',description:'Cell perimeter',method:'code',status:'retained',reusable:true});
      if(endpoint==='runs/ok' && onlyRetained)data.features[1].method='code';
      return {ok:true,json:async()=>data};
    }};
  vm.createContext(context);
  const root=path.join(__dirname,'../design-preview');
  for(const file of ['app.js','runtime.js'])vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),context);
  await flush();await flush();
  async function click(action,extra={}){
    let stopped=false;
    const button={dataset:{action,...extra},disabled:false};
    const event={target:{closest:()=>button},preventDefault(){},stopImmediatePropagation(){stopped=true;}};
    const handlers=element('app').handlers.filter(h=>h.type==='click').sort((a,b)=>Number(!!b.capture)-Number(!!a.capture));
    for(const h of handlers){h.fn(event);if(stopped)break;}
    for(let i=0;i<8;i++)await flush();
  }
  function change(id,checked){for(const h of element('app').handlers.filter(h=>h.type==='change'))h.fn({target:{id,checked,dataset:{}}});}
  async function pickFolder(id,files){
    let stopped=false;
    const event={target:{files,value:''},preventDefault(){},stopImmediatePropagation(){stopped=true;}};
    const handlers=element(id).handlers.filter(h=>h.type==='change').sort((a,b)=>Number(!!b.capture)-Number(!!a.capture));
    for(const h of handlers){h.fn(event);if(stopped)break;}
    for(let i=0;i<8;i++)await flush();
  }
  return {context,calls,element,click,change,pickFolder,run:code=>vm.runInContext(code,context)};
}

test('Visualize opens an independent CSV upload page and shows retained features by source',async()=>{
  const app=await ui();await app.click('navigate',{page:'visualize'});
  assert.match(app.run('visualizePage()'),/Upload feature_value.csv/);
  await app.click('visualize-upload');
  assert.equal(app.calls.find(c=>c.endpoint==='native-picker').initial,'/workspace/exports');
  const html=app.run('visualizePage()');
  assert.match(html,/Code/);assert.match(html,/VLM/);
  assert.ok(!html.includes('data-index="2"'));
  assert.match(html,/<svg[^>]*class="feature-histogram"/);
  for(const text of ['<table','Validation score','Validation decisions','Feature design and provenance','Visual context','Cell area','Per-sample'])assert.ok(!html.includes(text),text);
  assert.ok(!app.calls.some(c=>['runs','compute','settings'].includes(c.endpoint)));
  await app.click('live-viz',{index:'1'});
  assert.ok(app.calls.some(c=>c.endpoint==='runs/ok/distribution?feature=visual'));
});

test('Help uses the existing Model API and renders grounded Markdown safely',async()=>{
  const app=await ui();await app.click('navigate',{page:'help'});
  assert.match(app.run('helpPage()'),/Ask MorphAgent/);
  assert.match(app.run('helpPage()'),/I’m MorphAgent/);
  app.run("state.help.draft='What is MorphAgent?'");
  await app.click('help-send');
  assert.match(app.run('runDialog()'),/API key required/);
  app.run("state.runDialog=null;Object.assign(state.config,{baseUrl:'https://test.invalid/v1',apiKey:'typed',model:'paper-model'})");
  await app.click('help-send');
  const request=app.calls.find(c=>c.endpoint==='help/ask');
  assert.equal(request.body.question,'What is MorphAgent?');
  assert.deepEqual(request.body.history,[]);
  const html=app.run('helpPage()');
  assert.match(html,/<strong>Grounded<\/strong> answer \[Manuscript\]/);
  assert.ok(!html.includes('<script>'));
  assert.match(html,/&lt;script&gt;/);
  await app.click('help-clear');
  assert.match(app.run('helpPage()'),/Ask MorphAgent/);
  assert.equal(app.run('state.help.messages.length'),0);
});

test('No-knowledge Design confirmation offers optional Deep Research',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'dataset',summary:{sample_count:1}};state.question='Measure';state.featureNumber='5';state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  await app.click('prepare-run');
  assert.match(app.run('runDialog()'),/Deep Research/);
  app.change('deep-research-checkbox',true);
  await app.click('confirm-run');
  assert.equal(app.calls.find(c=>c.endpoint==='runs').body.deepResearch,true);
});

test('Attached knowledge and Deep Research can be combined',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'dataset',summary:{sample_count:1}};state.question='Measure';state.featureNumber='5';state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model';state.docs=[{id:'doc',name:'notes.pdf'}]");
  await app.click('prepare-run');
  assert.match(app.run('runDialog()'),/notes.pdf/);
  assert.ok(app.run('runDialog()').includes('deep-research-checkbox'));
  app.change('deep-research-checkbox',true);
  await app.click('confirm-run');
  const body=app.calls.find(c=>c.endpoint==='runs').body;
  assert.equal(body.deepResearch,true);
  assert.deepEqual(body.documentIds,['doc']);
});

test('Compute confirmation omits the execution explanation',async()=>{
  const app=await ui();
  await app.click('compute-source');await app.click('reuse-demo');
  app.run("state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  await app.click('prepare-compute');
  assert.ok(!app.run('runDialog()').includes('Uses the saved scripts unchanged'));
});

test('History Visualize works but new drafts do not inherit its results',async()=>{
  const app=await ui({historyKind:'discovery'});
  app.run("state.question='Unsaved question';state.featureNumber='12'");
  await app.click('load-job',{id:'ok'});
  await app.click('navigate',{page:'visualize'});
  assert.match(app.run('visualizePage()'),/feature-histogram/);
  await app.click('navigate',{page:'data'});
  await app.click('navigate',{page:'visualize'});
  assert.match(app.run('visualizePage()'),/Upload feature_value.csv/);
  assert.ok(!app.run('visualizePage()').includes('feature-histogram'));
  await app.click('visualize-upload');
  assert.equal(app.run('state.question'),'Unsaved question');
  assert.equal(app.run('state.featureNumber'),'12');
  await app.click('navigate',{page:'data'});
  assert.match(app.run('home()'),/id="question"/);
});

test('Canceling run selection keeps the current visualization',async()=>{
  const app=await ui();await app.click('navigate',{page:'visualize'});await app.click('visualize-upload');
  app.context.window.prompt=()=>null;
  await app.click('visualize-upload');
  assert.match(app.run('visualizePage()'),/feature-histogram/);
  assert.equal(app.calls.filter(c=>c.endpoint==='runs/load').length,1);
});

test('Visualize method filters show matching features and keep histogram indices correct',async()=>{
  const app=await ui({additionalCode:true});
  await app.click('navigate',{page:'visualize'});await app.click('visualize-upload');
  const selected=html=>[...html.matchAll(/class="viz-feature [^"]*" data-action="live-viz" data-index="(\d+)"/g)].map(m=>Number(m[1]));
  assert.match(app.run('visualizePage()'),/data-action="visualize-filter" data-value="all" aria-pressed="true"/);
  assert.deepEqual(selected(app.run('visualizePage()')),[0,1,3]);
  await app.click('visualize-filter',{value:'code'});
  assert.deepEqual(selected(app.run('visualizePage()')),[0,3]);
  await app.click('live-viz',{index:'3'});
  assert.equal(app.calls.at(-1).endpoint,'runs/ok/distribution?feature=perimeter');
  await app.click('visualize-filter',{value:'vlm'});
  assert.deepEqual(selected(app.run('visualizePage()')),[1]);
  assert.equal(app.run('state.vizFeature'),1);
  assert.equal(app.calls.at(-1).endpoint,'runs/ok/distribution?feature=visual');
  assert.match(app.run('visualizePage()'),/data-value="vlm" aria-pressed="true"/);
  const requests=app.calls.length;
  await app.click('visualize-filter',{value:'all'});
  assert.deepEqual(selected(app.run('visualizePage()')),[0,1,3]);
  assert.equal(app.run('state.vizFeature'),1);
  assert.equal(app.calls.length,requests); // Keep the still-visible histogram.
});

test('Empty method filter clears stale chart and importing a run resets to All',async()=>{
  const app=await ui({onlyRetained:true});
  await app.click('navigate',{page:'visualize'});await app.click('visualize-upload');
  await app.click('visualize-filter',{value:'vlm'});
  assert.match(app.run('visualizePage()'),/No VLM features/);
  assert.ok(!app.run('visualizePage()').includes('feature-histogram'));
  assert.ok(!app.run('visualizePage()').includes('data-action="live-viz"'));
  assert.equal(app.run('state.vizFeature'),-1);
  await app.click('visualize-filter',{value:'code'});
  assert.match(app.run('visualizePage()'),/feature-histogram/);
  await app.click('visualize-filter',{value:'vlm'});
  await app.click('visualize-upload');
  assert.match(app.run('visualizePage()'),/data-value="all" aria-pressed="true"/);
  assert.match(app.run('visualizePage()'),/feature-histogram/);
});

test('Uploading a visualization does not replace an active Design run',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'dataset',summary:{sample_count:1}};state.question='Measure';state.featureNumber='5';state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  await app.click('prepare-run');await app.click('confirm-run');
  await app.click('navigate',{page:'visualize'});await app.click('visualize-upload');
  assert.match(app.run('visualizePage()'),/feature-histogram/);
  await app.click('navigate',{page:'data'});
  assert.match(app.run('home()'),/submitted-question/);
  assert.match(app.run('home()'),/Measure/);
  assert.equal(app.calls.filter(c=>c.endpoint==='runs').length,1);
  assert.ok(!app.calls.some(c=>c.endpoint.endsWith('/cancel')));
});

test('Model settings retain required inputs without extra explanatory copy',async()=>{
  const app=await ui();const html=app.run('apiSettings()+settings()');
  assert.ok(!html.includes('Use free demo API'));
  assert.ok(!html.includes('local .env file'));
  for(const key of ['baseUrl','apiKey','model'])assert.match(html,new RegExp('data-config="'+key+'"'));
  for(const text of ['Model connection</h2>','Connect an OpenAI-compatible','Your own API connection','API keys stay','Your models, your workflow','Preview only. Credentials'])assert.ok(!html.includes(text),text);
  assert.match(html,/data-action="save-settings"/);
});

test('History shows actual tasks without the demo shortcut or imported browsing records',async()=>{
  const app=await ui({includeImports:true});
  const html=app.run('sidebar()');
  assert.match(html,/data-action="load-job" data-id="ok"/);
  assert.match(html,/data-action="load-job" data-id="bad"/);
  assert.match(html,/Completed/);assert.match(html,/Failed/);
  for(const text of ['live-example','Browse the completed','Bundled results','imported-results','completed_demo_run','legacy-loaded-run','Imported · unverified'])assert.ok(!html.includes(text),text);
  assert.ok(!app.calls.some(c=>c.endpoint.endsWith('/remove')));
});

test('Only imported runs leave History empty while upload remains available',async()=>{
  const app=await ui({onlyImports:true});
  assert.match(app.run('sidebar()'),/class="history-empty"/);
  assert.ok(!app.run('sidebar()').includes('data-action="load-job"'));
  await app.click('navigate',{page:'visualize'});
  assert.match(app.run('visualizePage()'),/Upload feature_value.csv/);
  await app.click('navigate',{page:'compute'});
  assert.match(app.run('reusePage()'),/Upload features/);
});

test('Running Design keeps the ETA but removes its explanatory footer',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'dataset',summary:{sample_count:1}};state.question='Measure';state.featureNumber='5';state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  await app.click('prepare-run');await app.click('confirm-run');
  const html=app.run('runPage()');
  assert.match(html,/Estimated remaining/);assert.match(html,/About/);
  assert.match(html,/Stop run/);assert.match(html,/Live output/);
  assert.ok(!html.includes('Approximate · updates with progress'));
  assert.ok(!html.includes('API latency and code retries'));
});
test('Design and Compute omit the top bar and helper footers',async()=>{
  const app=await ui();
  const html=app.run('header()+sidebar()+home()+reusePage()');
  for(const text of ['class="topbar"','New analysis','Local runtime','Local workspace','sidebar-footnote','composer-hint','home-bottom','Enter to review configuration','Model &amp; analysis in Settings','Saved scripts run locally'])assert.ok(!html.includes(text),text);
  assert.match(app.run('header()'),/aria-label="Open sidebar"/);
  assert.match(html,/id="question"/);
  // Compute replays a fixed feature set, so it offers attachments instead of a prompt.
  assert.ok(!app.run('reusePage()').includes('<textarea'));
  assert.match(html,/data-action="compute-source"/);
});
test('Analysis settings keep choices and loop counts without section explanations',async()=>{
  const app=await ui();app.run("state.settingsTab='analysis'");
  const html=app.run('settings()');
  for(const text of ['Choose how MorphAgent','Each loop plans','Reproducibility is enabled','API keys stay','settings-footer"><small>'])assert.ok(!html.includes(text),text);
  for(const value of ['both','code','vlm','ultra','fast','detailed'])assert.match(html,new RegExp('data-value="'+value+'"'));
  for(const count of ['1 loop','5 loops','20 loops'])assert.ok(html.includes(count));
});
test('Data submit waits for configuration confirmation, then launches only once',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'dataset',summary:{sample_count:1}};state.question='Measure';state.featureNumber='17';state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  await app.click('prepare-run');
  assert.equal(app.calls.filter(c=>c.endpoint==='runs').length,0);
  assert.equal(app.run('state.page'),'data');
  assert.match(app.run('runDialog()'),/Confirm and run/);
  await Promise.all([app.click('confirm-run'),app.click('confirm-run')]);
  await app.click('confirm-run');
  assert.equal(app.calls.filter(c=>c.endpoint==='runs').length,1);
  assert.equal(app.calls.find(c=>c.endpoint==='runs').body.featureNumber,17);
  assert.equal(app.run('state.page'),'data');
  assert.ok(!app.run('runPage()').includes('Start analysis'));
  assert.match(app.run('home()'),/submitted-question/);
  assert.match(app.run('home()'),/Estimated remaining/);
  assert.ok(!app.run('home()').includes('id="question"'));
});
test('Compute displays all available code without selection controls',async()=>{
  const app=await ui();let html=app.run('reusePage()');
  assert.match(html,/Upload features/);
  assert.ok(!html.includes('<textarea'));
  assert.ok(!html.includes('failed-run'));
  await app.click('compute-source');
  assert.equal(app.calls.find(c=>c.endpoint==='native-picker').initial,'/workspace/exports');
  await app.click('compute-features');
  html=app.run('reusePage()');
  assert.match(html,/Cell area/);
  assert.match(html,/Saved features/);
  assert.ok(!html.includes('type="checkbox"'));
  assert.ok(!html.includes('reuse-all')&&!html.includes('reuse-clear'));
  assert.match(html,/Add data/);
  assert.match(html,/Replays saved code/);
  assert.match(html,/Rescored by the VLM/);
  assert.match(html,/Nothing reusable saved/);
});
test('Design, Compute, Visualize and Help navigation live only in the sidebar',async()=>{
  const app=await ui();
  assert.match(app.run('sidebar()'),/Completed/);
  assert.match(app.run('sidebar()'),/Failed/);
  const html=app.run('sidebar()'),header=app.run('header()');
  for(const [id,name] of [['data','Design'],['compute','Compute'],['visualize','Visualize'],['help','Help']]) {
    assert.ok(html.includes('data-page="'+id+'"'));
    assert.ok(html.includes('>'+name+'<'));
  }
  for(const text of ['New run','Compute features','Load previous run','Feature design'])assert.ok(!html.includes(text));
  assert.ok(!header.includes('data-page=')&&!header.includes('Analysis workflow'));
  assert.match(html,/data-action="delete-history"/);
});
test('Reuse monitor describes saved features without claiming new seeding or validation',async()=>{
  const app=await ui();
  await app.click('load-job',{id:'ok'});
  const html=app.run('runPage()');
  assert.match(html,/Saved features only/);
  assert.match(html,/Code is replayed, VLM features are rescored/);
  assert.ok(!html.includes('seed 42'));
  assert.match(html,/no new validation/i);
});

test('Knowledge is attached beside data, not in Settings',async()=>{
  const app=await ui();
  assert.match(app.run('home()'),/data-action="manage-knowledge"/);
  assert.match(app.run('home()'),/Upload knowledge/);
  assert.ok(!app.run('settings()').includes('data-tab="knowledge"'));
  assert.ok(!app.run('settings()').includes('Use external knowledge'));
});

test('Upload, inspect, add and remove references updates the composer label',async()=>{
  const app=await ui();
  await app.click('manage-knowledge');
  assert.equal(app.run('state.knowledge.open'),true);
  app.context.reference=Object.assign(new Blob(['Focus on axons.']),{name:'notes.txt'});
  await app.run('addDocs([reference])');
  assert.match(app.run('home()'),/Knowledge attached/);
  assert.match(app.run('knowledgeDialog()'),/notes.txt/);
  await app.click('preview-doc',{id:'doc0'});
  assert.match(app.run('knowledgeDialog()'),/Extracted scientific reference/);
  assert.match(app.run('knowledgeDialog()'),/Download original/);
  await app.click('knowledge-list');
  app.context.reference=Object.assign(new Blob(['More context.']),{name:'second.txt'});
  await app.run('addDocs([reference])');
  assert.equal(app.run('state.docs.length'),2);
  await app.click('remove-doc',{id:'doc0'});
  assert.match(app.run('home()'),/Knowledge attached/);
  await app.click('remove-doc',{id:'doc1'});
  assert.match(app.run('home()'),/Upload knowledge/);
  assert.ok(!app.run('home()').includes('Knowledge attached'));
  assert.match(app.run('knowledgeDialog()'),/No knowledge attached/);
});

test('Attached documents are sent even if old Settings disabled knowledge',async()=>{
  const app=await ui();
  app.run("state.docs=[{id:'doc0',name:'notes.txt',size:20}];state.dataset={id:'d',name:'Dataset',summary:{sample_count:1,primary_image_count:1,mask_count:0}};state.question='Measure';state.config.knowledgeEnabled=false;state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  app.run("state.featureNumber='5'");
  await app.click('prepare-run');
  await app.click('confirm-run');
  const call=app.calls.find(c=>c.endpoint==='runs');
  assert.equal(call.body.knowledgeEnabled,true);
  assert.deepEqual(call.body.documentIds,['doc0']);
});

test('No attachments means no hidden external-knowledge selection',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'Dataset',summary:{sample_count:1,primary_image_count:1,mask_count:0}};state.question='Measure';state.config.knowledgeEnabled=true;state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  app.run("state.featureNumber='5'");
  await app.click('prepare-run');
  await app.click('confirm-run');
  assert.equal(app.calls.find(c=>c.endpoint==='runs').body.knowledgeEnabled,false);
});

test('Partial upload failures keep successful files visible and show the error',async()=>{
  const app=await ui({failDocumentName:'broken.pdf'});
  app.context.references=[Object.assign(new Blob(['Good.']),{name:'good.txt'}),Object.assign(new Blob(['bad']),{name:'broken.pdf'})];
  await app.run('addDocs(references)');
  assert.equal(app.run('state.docs.length'),1);
  assert.equal(app.run('state.knowledge.uploading'),false);
  assert.match(app.run('knowledgeDialog()'),/good.txt/);
  assert.match(app.run('knowledgeDialog()'),/broken.pdf/);
  assert.match(app.run('knowledgeDialog()'),/No readable text/);
});

test('Feature number is blank and required, with a blocking modal for invalid values',async()=>{
  const app=await ui();
  assert.equal(app.run('state.featureNumber'),'');
  assert.match(app.run('home()'),/id="feature-number"[^>]*required/);
  app.run("state.dataset={id:'d',name:'Dataset',summary:{sample_count:1,primary_image_count:1,mask_count:0}};state.question='Measure';state.config.apiKey='typed';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  for(const value of ['', '0', '-2', '2.5', 'abc', '501']){
    app.context.count=value;app.run('state.featureNumber=count');
    await app.click('prepare-run');
    assert.match(app.run('runDialog()'),/role="alertdialog"/);
    assert.match(app.run('runDialog()'),/Feature number required|Invalid feature number/);
    assert.equal(app.calls.filter(c=>c.endpoint==='runs').length,0);
    await app.click('close-run-dialog');
  }
});

test('Missing API key has a prominent dialog and a shortcut to Model API',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'Dataset',summary:{sample_count:1,primary_image_count:1,mask_count:0}};state.question='Measure';state.featureNumber='5';state.config.baseUrl='https://test.invalid';state.config.model='model'");
  await app.click('prepare-run');
  assert.match(app.run('runDialog()'),/API key required/);
  assert.match(app.run('runDialog()'),/Open Model API/);
  assert.equal(app.calls.filter(c=>c.endpoint==='runs').length,0);
  await app.click('resolve-run-issue');
  assert.equal(app.run('state.page'),'settings');
  assert.equal(app.run('state.runDialog'),null);
});

test('Separate VLM credentials must also be supplied',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'Dataset',summary:{sample_count:1,primary_image_count:1,mask_count:0}};state.question='Measure';state.featureNumber='5';Object.assign(state.config,{apiKey:'typed',baseUrl:'https://test.invalid',model:'model',route:'both',sameConnection:false,vlmBaseUrl:'https://vision.invalid',vlmModel:'vision'})");
  await app.click('prepare-run');
  assert.match(app.run('runDialog()'),/VLM API key required/);
  assert.equal(app.calls.filter(c=>c.endpoint==='runs').length,0);
});

test('Configuration lists all reviewed inputs, excludes keys, and cancels without launching',async()=>{
  const app=await ui();
  app.run("state.dataset={id:'d',name:'My cells',path:'/local/cells',summary:{sample_count:20,primary_image_count:20}};state.question='Measure <script>alert(1)</script>';state.featureNumber='23';state.docs=[{id:'ref',name:'Tau paper.pdf'}];Object.assign(state.config,{apiKey:'never-show-secret',baseUrl:'https://test.invalid',model:'model',mode:'fast'})");
  await app.click('prepare-run');
  const html=app.run('runDialog()');
  for(const text of ['My cells','Tau paper.pdf','Fast','5 loops','23','Feature number','Knowledge','Biological question'])assert.ok(html.includes(text),text);
  assert.ok(!html.includes('never-show-secret'));
  assert.ok(!html.includes('<script>'));
  await app.click('close-run-dialog');
  assert.equal(app.calls.filter(c=>c.endpoint==='runs').length,0);
  assert.equal(app.run('state.featureNumber'),'23');
  await app.click('prepare-run');
  app.run("state.featureNumber='40';state.question='Changed outside modal'");
  await app.click('confirm-run');
  const request=app.calls.find(c=>c.endpoint==='runs').body;
  assert.equal(request.featureNumber,23);
  assert.equal(request.question,'Measure <script>alert(1)</script>');
});

test('Applied keys remain visibly masked without putting the secret back into HTML',async()=>{
  const app=await ui();
  app.run("Object.assign(state.config,{apiKey:'do-not-echo',baseUrl:'https://test.invalid',model:'model'})");
  await app.click('save-settings');
  const html=app.run('apiSettings()');
  assert.match(html,/value="\*{8}"/);
  assert.match(html,/readonly/);
  assert.ok(!html.includes('do-not-echo'));
  assert.equal(app.run('state.config.apiKey'),'');
  await app.click('save-settings');
  assert.equal(app.calls.filter(c=>c.endpoint==='settings').at(-1).body.apiKey,'');
  await app.click('edit-api-key',{key:'apiKey'});
  assert.ok(!app.run('apiSettings()').includes('value="********"'));
});

test('Design combines input and progress and exposes automatic download',async()=>{
  const app=await ui();
  const header=app.run('header()');
  assert.match(app.run('sidebar()'),/>Design</);
  assert.ok(!header.includes('data-page="run"'));
  await app.click('load-job',{id:'ok'});
  assert.equal(app.run('state.page'),'compute');
  const html=app.run('reusePage()');
  assert.match(html,/Results saved/);
  assert.match(html,/20260915_120000_000000/);
  assert.match(html,/data-action="download-bundle"/);
});

test('History deletion confirms, cancels, and removes only the intended entry',async()=>{
  const app=await ui();
  await app.click('load-job',{id:'ok'});
  await app.click('delete-history',{id:'ok'});
  assert.match(app.run('runDialog()'),/Delete run/);
  assert.match(app.run('runDialog()'),/20260915_120000_000000/);
  assert.match(app.run('runDialog()'),/system Trash/i);
  assert.ok(!app.run('runDialog()').includes('.web_workspace/trash'));
  assert.match(app.run('runDialog()'),/\/workspace\/runs\/20260915_120000_000000/);
  assert.match(app.run('runDialog()'),/\.zip/);
  assert.equal(app.calls.filter(c=>c.endpoint==='runs/ok/remove').length,0);
  await app.click('close-run-dialog');
  assert.match(app.run('reusePage()'),/Results saved/);
  await app.click('delete-history',{id:'ok'});
  await Promise.all([app.click('confirm-delete-history'),app.click('confirm-delete-history')]);
  assert.equal(app.calls.filter(c=>c.endpoint==='runs/ok/remove').length,1);
  assert.ok(!app.run('sidebar()').includes('data-id="ok"'));
  assert.ok(app.run('sidebar()').includes('data-id="bad"'));
  assert.ok(!app.run('reusePage()').includes('Results saved'));
  assert.equal(app.run('state.runDialog'),null);
});

test('Completed Design monitor has no redundant New design button',async()=>{
  const app=await ui({historyKind:'discovery'});
  await app.click('load-job',{id:'ok'});
  assert.ok(!app.run('home()').includes('New design'));
  assert.ok(!app.run('home()').includes('data-action="live-new"'));
  assert.match(app.run('home()'),/View results/);
  await app.click('navigate',{page:'data'});
  assert.match(app.run('home()'),/id="question"/);
});

test('Compute blocks missing API, confirms all inputs, and stays on Compute after launch',async()=>{
  const app=await ui();
  await app.click('compute-source');
  await app.click('reuse-demo');
  app.run("state.question='Independent design question';");
  await app.click('prepare-compute');
  assert.match(app.run('runDialog()'),/API key required/);
  assert.ok(!app.calls.some(c=>c.endpoint==='compute'));
  await app.click('close-run-dialog');
  app.run("Object.assign(state.config,{baseUrl:'https://test.invalid',model:'test',apiKey:'typed-secret'})");
  await app.click('prepare-compute');
  const confirmation=app.run('runDialog()');
  // The question is inherited from the source run rather than asked for again.
  for(const text of ['Historical question','Previous run','20260915_120000_000000','New dataset','area'])assert.ok(confirmation.includes(text));
  assert.ok(!confirmation.includes('typed-secret'));
  assert.ok(!app.calls.some(c=>c.endpoint==='compute'));
  await app.click('close-run-dialog');
  await app.click('prepare-compute');
  await Promise.all([app.click('confirm-run'),app.click('confirm-run')]);
  assert.equal(app.calls.filter(c=>c.endpoint==='compute').length,1);
  assert.equal(app.calls.find(c=>c.endpoint==='compute').body.question,undefined);
  assert.deepEqual(app.calls.find(c=>c.endpoint==='compute').body.featureNames,['area','visual']);
  assert.equal(app.run('state.page'),'compute');
  assert.match(app.run('reusePage()'),/submitted-question/);
  assert.match(app.run('reusePage()'),/Compute cell area/);
  assert.equal(app.run('state.question'),'Independent design question');
});

test('Filtering the read-only feature list still computes every reusable feature',async()=>{
  const app=await ui({additionalCode:true});
  await app.click('compute-source');await app.click('reuse-demo');
  app.run("state.search='perimeter';Object.assign(state.config,{apiKey:'fixture',baseUrl:'https://test.invalid',model:'fixture'})");
  assert.ok(!app.run('featureRows()').includes('Cell area'));
  assert.match(app.run('featureRows()'),/Cell perimeter/);
  await app.click('prepare-compute');
  assert.match(app.run('runDialog()'),/area/);
  assert.match(app.run('runDialog()'),/perimeter/);
  await app.click('confirm-run');
  // Saved VLM features are rescored alongside the saved code, never silently dropped.
  assert.deepEqual(app.calls.find(c=>c.endpoint==='compute').body.featureNames,['area','visual','perimeter']);
});

for(const historyKind of ['discovery','reuse'])test(`Leaving ${historyKind} history opens independent new-run forms`,async()=>{
  const app=await ui({historyKind});
  await app.click('load-job',{id:'ok'});
  assert.match(app.run('runPage()'),/Historical question/);
  await app.click('navigate',{page:'data'});
  assert.match(app.run('home()'),/id="question"/);
  assert.ok(!app.run('home()').includes('submitted-question'));
  assert.equal(app.run('state.question'),'');
  assert.equal(app.run('state.dataset'),null);
  assert.equal(app.run('state.featureNumber'),'');
  await app.click('navigate',{page:'compute'});
  assert.match(app.run('reusePage()'),/Compute saved features/);
  assert.ok(!app.run('reusePage()').includes('submitted-question'));
  assert.ok(!app.run('sidebar()').includes('history-entry active'));
  assert.ok(!app.calls.some(c=>c.endpoint.endsWith('/cancel')||c.endpoint.endsWith('/remove')));
});

test('History inspection preserves both unsubmitted drafts and Compute attachments',async()=>{
  const app=await ui({historyKind:'discovery'});
  await app.click('compute-source');await app.click('reuse-demo');
  app.run("state.question='New design draft';state.featureNumber='8';state.dataset={id:'draft',name:'Draft images',summary:{sample_count:2}}");
  await app.click('load-job',{id:'ok'});
  await app.click('navigate',{page:'data'});
  assert.match(app.run('home()'),/id="question"/);
  assert.equal(app.run('state.question'),'New design draft');
  assert.equal(app.run('state.dataset.id'),'draft');
  assert.equal(app.run('state.featureNumber'),'8');
  await app.click('navigate',{page:'compute'});
  assert.match(app.run('reusePage()'),/Previous run attached/);
  assert.match(app.run('reusePage()'),/Data attached/);
});

test('Design navigation returns to its active run after inspecting history without cancelling it',async()=>{
  const app=await ui();
  app.run("state.question='Measure';state.featureNumber='5';state.dataset={id:'draft',name:'Data',summary:{sample_count:1}};Object.assign(state.config,{apiKey:'fixture',baseUrl:'https://test.invalid',model:'fixture'})");
  await app.click('prepare-run');await app.click('confirm-run');
  await app.click('load-job',{id:'ok'});
  await app.click('navigate',{page:'data'});
  assert.match(app.run('home()'),/submitted-question/);
  assert.match(app.run('home()'),/Estimated remaining/);
  assert.ok(!app.run('home()').includes('Historical question'));
  await app.click('navigate',{page:'compute'});
  assert.match(app.run('reusePage()'),/Compute saved features/);
  await app.click('navigate',{page:'data'});
  assert.match(app.run('home()'),/Estimated remaining/);
  assert.equal(app.calls.filter(c=>c.endpoint==='runs').length,1);
  assert.ok(!app.calls.some(c=>c.endpoint.endsWith('/cancel')));
});

test('Compute navigation restores its active run without consuming the Design draft',async()=>{
  const app=await ui({historyKind:'discovery'});
  await app.click('compute-source');await app.click('reuse-demo');
  app.run("state.question='Unsubmitted design';state.featureNumber='9';Object.assign(state.config,{apiKey:'fixture',baseUrl:'https://test.invalid',model:'fixture'})");
  await app.click('prepare-compute');await app.click('confirm-run');
  assert.equal(app.run('state.question'),'Unsubmitted design');
  await app.click('load-job',{id:'ok'});
  await app.click('navigate',{page:'compute'});
  assert.match(app.run('reusePage()'),/Estimated remaining/);
  assert.match(app.run('reusePage()'),/Compute cell area/);
  await app.click('navigate',{page:'data'});
  assert.match(app.run('home()'),/id="question"/);
  assert.equal(app.run('state.question'),'Unsubmitted design');
  assert.equal(app.run('state.featureNumber'),'9');
  await app.click('navigate',{page:'compute'});
  assert.match(app.run('reusePage()'),/Estimated remaining/);
  assert.equal(app.calls.filter(c=>c.endpoint==='compute').length,1);
  assert.ok(!app.calls.some(c=>c.endpoint.endsWith('/cancel')));
});

const fakeFiles = (count,size=1024) => Array.from({length:count},(_,i)=>(
  {name:'image.tif',size,webkitRelativePath:`dataset/sample_${i}/image.tif`}));

test('A dataset too large to copy is refused with the alternative named',async()=>{
  const app=await ui();
  await app.pickFolder('dataset-input',fakeFiles(5000));
  const shown=app.run('home()');
  assert.match(shown,/5,000 files/);
  assert.match(shown,/Use &quot;Paste folder path&quot; instead/);
  // Nothing may be copied before the refusal.
  assert.ok(!app.calls.some(c=>c.endpoint.startsWith('imports')));
  assert.equal(app.run('state.dataset === null'),true);
});

test('A dataset folder is read in place instead of being copied',async()=>{
  const app=await ui();
  assert.match(app.run('home()'),/data-action="data-path"/);
  await app.click('data-path');
  assert.equal(app.calls.find(c=>c.endpoint==='datasets').body.path,'/workspace/exports/20260915_120000_000000');
  assert.equal(app.run('state.dataset.path'),'/workspace/exports/20260915_120000_000000');
  assert.ok(!app.calls.some(c=>c.endpoint.startsWith('imports')));
});

test('Compute offers the same in-place folder route for its target dataset',async()=>{
  const app=await ui();await app.click('navigate',{page:'compute'});
  assert.match(app.run('reusePage()'),/data-action="reuse-path"/);
  await app.click('reuse-path');
  assert.equal(app.calls.find(c=>c.endpoint==='datasets').body.path,'/workspace/exports/20260915_120000_000000');
  assert.ok(!app.calls.some(c=>c.endpoint.startsWith('imports')));
});

test('A small folder still uploads, reporting progress without a toast per file',async()=>{
  const app=await ui();
  await app.pickFolder('dataset-input',fakeFiles(3));
  const puts=app.calls.filter(c=>c.endpoint.startsWith('imports/up1?name='));
  assert.equal(puts.length,3);
  assert.equal(app.run('state.dataset.path'),'/workspace/.web_workspace/imports/up1');
  assert.ok(!app.run('home()').includes('too many to copy'));
});
