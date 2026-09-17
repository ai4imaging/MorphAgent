/* Real local runtime. Static previews remain usable without launch_web_ui.py. */
(() => {
  'use strict';
  const token = document.querySelector('meta[name="morphagent-token"]')?.content;
  if (!token) return;
  const live = {job:null,runs:[],datasets:[],evidence:null,logs:[],offset:0,busy:false,connected:true,preview:null,objectUrls:[],reuseSource:null,reuseData:null,lastError:'',launching:false};
  const active = j => j && ['starting','running','cancelling'].includes(j.status);
  live.designDraft=true;
  live.computeDraft=true;
  live.exportsDirectory='';
  live.visualizeJob=null;
  live.visualizeFilter='all';
  live.distribution=null;
  live.distributionError='';
  const esc = escapeHTML;
  const base = {home,settings,apiSettings,analysisSettings,datasetCard};
  const buttonHTML = (label,action,primary=false,attrs='') => button(label,action,'',primary,attrs);
  const cards = () => live.job?.features || [];
  const visualCards = () => live.visualizeJob?.features || [];
  const visibleFeatures = () => visualCards().map((feature,index)=>({feature,index}))
    .filter(({feature})=>feature.status!=='dropped' && (live.visualizeFilter==='all'||feature.method===live.visualizeFilter));
  // A selected result is independent of each workflow's unsubmitted draft.
  const showingRun = () => !!live.job && (state.page==='data'
    ? live.job.kind!=='reuse' && !live.designDraft
    : state.page==='compute' && live.job.kind==='reuse' && !live.computeDraft);
  const statusLabel = j => ({complete:'Completed',failed:'Failed',partial:'Partially completed',empty:'No results',cancelled:'Stopped',interrupted:'Interrupted',loaded:'Imported · unverified',starting:'Starting',running:'Running',cancelling:'Stopping'}[j?.status]||'Not started');
  const statusIcon = j => `<span class="run-state-icon" title="${statusLabel(j)}" aria-label="${statusLabel(j)}">${icon(active(j)?'spinner':j?.status==='complete'?'check':j?.status==='loaded'?'folder':j?'close':'clock',active(j)?'spinner':'')}</span>`;
  const displayName = j => esc(j?.timestamp ? j.timestamp.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2}).*$/,'$1-$2-$3 $4:$5:$6') : j?.name||'');
  const empty = text => `<div class="empty">${esc(text)}</div>`;
  const elapsed = () => {
    if (!live.job) return '';
    const seconds = Math.max(0,Math.floor((live.job.finishedAt || Date.now()/1000) - live.job.startedAt));
    return `${Math.floor(seconds/60)}m ${seconds%60}s`;
  };
  async function api(path,options={}) {
    const headers = {'X-MorphAgent-Token':token,...options.headers};
    if (options.body && !(options.body instanceof Blob)) {
      headers['Content-Type']='application/json';options.body=JSON.stringify(options.body);
    }
    const response=await fetch('/api/'+path,{...options,headers});
    if (!response.ok) { const error=await response.json();throw new Error(error.error || 'Request failed.'); }
    return response;
  }
  const get = async path => (await api(path)).json();
  const post = async (path,body={}) => (await api(path,{method:'POST',body})).json();
  function payload() {return {datasetId:state.dataset?.id,question:state.question.trim(),featureNumber:Number(state.featureNumber),route:state.config.route,mode:state.config.mode,knowledgeEnabled:state.docs.length>0,documentIds:state.docs.map(d=>d.id),deepResearch:false};}
  function setSettings(c) {Object.assign(state.config,c,{apiKey:'',vlmApiKey:''});state.editingSecrets.clear();}
  async function refreshHistory() {const data=await get('bootstrap');live.runs=data.runs;live.datasets=data.datasets;live.exportsDirectory=data.exportsDirectory||'';return data;}
  async function saveSettings() {
    state.savingSettings=true;render();
    try {setSettings(await post('settings',state.config));toast('Settings applied.');}
    finally {state.savingSettings=false;render();}
  }
  async function chooseDemo() {
    state.dataset=await post('datasets/demo');
    if (!state.question) state.question='Generate unbiased morphological features that quantify Tau protein aggregation and neuronal structure in these images.';
    render();toast(`${state.dataset.summary.sample_count} demo samples ready.`);
  }
  async function askHelp() {
    if(state.help.pending)return;
    const question=state.help.draft.trim();
    if(!question){state.help.error='Enter a question about the paper.';render();return;}
    if(question.length>4000){state.help.error='Keep your question under 4,000 characters.';render();return;}
    if(!state.config.apiKey.trim()&&!state.config.hasApiKey)return showRunIssue({title:'API key required',message:'Set your Model API connection before asking about the paper.',target:'api',label:'Open Model API'});
    if(!state.config.baseUrl.trim()||!state.config.model.trim())return showRunIssue({title:'Model connection incomplete',message:'Fill in Base URL and Model in Settings.',target:'api',label:'Open Model API'});
    if(state.config.apiKey.trim())setSettings(await post('settings',state.config));
    const history=state.help.messages.slice(-8).map(({role,content})=>({role,content}));
    state.help.messages.push({role:'user',content:question});
    state.help.draft='';state.help.error='';state.help.pending=true;render();
    try {
      const result=await post('help/ask',{question,history});
      state.help.messages.push({role:'assistant',content:result.answer});
    } catch(e) {
      state.help.messages.pop();state.help.draft=question;state.help.error=e.message;
    } finally {state.help.pending=false;render();document.querySelector('.help-thread')?.scrollTo?.(0,1000000);}
  }
  sendHelp=()=>askHelp().catch(error);
  async function loadJob(id,page=null) {
    const job=await get('runs/'+id);
    if(page==='visualize')return showVisualization(job);
    live.job=job;live.logs=[];live.offset=0;live.evidence=null;live.preview=null;
    page=page||(live.job.kind==='reuse'?'compute':'data');
    if(live.job.kind==='reuse')live.computeDraft=false;
    else live.designDraft=false;
    state.historyOpen=true;state.vizFeature=0;state.selectedFeatures.clear();
    await updateLogs();navigate(page);
  }
  async function openWorkflow(page) {
    if(page==='data' || page==='compute') {
      await refreshHistory();
      const running=live.runs.find(j=>active(j) && (j.kind==='reuse')===(page==='compute'));
      if(running)return loadJob(running.id,page);
      if(page==='compute')live.computeDraft=true;
      else live.designDraft=true;
      state.historyOpen=false;
    } else if(page==='visualize') {
      if(state.page==='visualize')return;
      return showVisualization(state.historyOpen ? live.job : null);
    }
    navigate(page);
  }
  async function updateLogs() {
    if(!live.job) return;
    const key=live.job.id;
    const log=await get(`runs/${key}/logs?offset=${live.offset}`);
    if(live.job?.id!==key)return;
    live.offset=log.offset;live.logs.push(...log.lines);live.logs=live.logs.slice(-2000);
  }
  async function showVisualization(job,index=0) {
    live.visualizeJob=job;live.visualizeFilter='all';live.distribution=null;live.distributionError='';state.vizFeature=index;
    navigate('visualize');
    const first=visibleFeatures().find(item=>item.index===index)||visibleFeatures()[0];
    if(first)await loadDistribution(first.index);
    else {state.vizFeature=-1;render();}
  }
  async function filterVisualization(value) {
    if(!['all','code','vlm'].includes(value)||value===live.visualizeFilter)return;
    live.visualizeFilter=value;
    const visible=visibleFeatures();
    if(visible.some(item=>item.index===state.vizFeature)){render();return;}
    if(visible.length)return loadDistribution(visible[0].index);
    state.vizFeature=-1;live.distribution=null;live.distributionError='';render();
  }
  async function loadDistribution(index) {
    const card=visualCards()[index]; if(!card)return;
    const key=live.visualizeJob.id;
    state.vizFeature=index;live.distribution=null;live.distributionError='';render();
    try {
      const data=await get(`runs/${key}/distribution?feature=${encodeURIComponent(card.name)}`);
      if(live.visualizeJob?.id!==key || visualCards()[state.vizFeature]?.name!==card.name)return;
      live.distribution=data;
    } catch(e) {
      if(live.visualizeJob?.id!==key || visualCards()[state.vizFeature]?.name!==card.name)return;
      live.distributionError=e.message;
    }
    render();
  }
  async function previewArtifact(path,type) {
    const key=live.job.id;
    const response=await api(`runs/${key}/artifact?path=${encodeURIComponent(path)}`);
    if(live.job?.id!==key)return;
    if(type==='image') {
      live.objectUrls.forEach(URL.revokeObjectURL);live.objectUrls=[];
      const url=URL.createObjectURL(await response.blob());live.objectUrls.push(url);
      live.preview={name:path,type,url};
    } else {live.preview={name:path,type:'text',text:(await response.text()).slice(0,100000)};}
    render();
  }
  async function downloadArtifact(path) {
    const response=await api(`runs/${live.job.id}/artifact?download=1&path=${encodeURIComponent(path)}`);
    const url=URL.createObjectURL(await response.blob());
    const a=document.createElement('a');a.href=url;a.download=path.split('/').pop();a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  const error = e => {live.lastError=e.message;toast(e.message);const target=document.getElementById('runtime-error');if(target)target.textContent=e.message;};
  async function task(work) {
    if(live.busy) return;
    live.busy=true;live.lastError='';
    try {await work();} catch(e) {live.launching=false;live.lastError=e.message;if(state.knowledge.open)state.knowledge.error=e.message;render();error(e);} finally {live.busy=false;live.launching=false;}
  }

  sidebar = function() {
    // Imported folders remain usable in Compute/Visualize, but are not new tasks.
    const history=live.runs.filter(j=>j.status!=='loaded'&&j.kind!=='saved');
    return `<aside class="sidebar ${state.sidebarOpen?'open':''}" aria-label="Workspace sidebar"><div class="brand">${mark('brand-mark')}<span>MorphAgent</span><button class="icon-button sidebar-close" data-action="sidebar" aria-label="Close sidebar">${icon('close')}</button></div>${workflowNavigation()}<div class="history"><div class="section-eyebrow"><span>History</span>${icon('history','small-icon')}</div>${history.map(j=>`<div class="history-entry ${state.historyOpen && live.job?.id===j.id?'active':''}"><button class="history-item state-history" data-action="load-job" data-id="${esc(j.id)}" title="${esc(j.name)}">${statusIcon(j)}<span class="history-copy"><span>${displayName(j)}</span><small>${statusLabel(j)}${j.kind==='reuse'?' · Compute':''}</small></span></button><button class="icon-button history-delete" data-action="delete-history" data-id="${esc(j.id)}" aria-label="Remove ${esc(j.name)} from History" title="${active(j)||j.exportStatus==='saving'?'Wait until this run finishes saving':'Remove from History'}" ${active(j)||j.exportStatus==='saving'?'disabled':''}>${icon('trash')}</button></div>`).join('')||'<p class="history-empty">Your analyses will appear here.</p>'}</div><div class="sidebar-bottom"><button class="nav-action ${state.page==='settings'?'active':''}" data-action="settings">${icon('settings')}<span>Settings</span></button></div></aside>`;
  };
  home = () => live.job && live.job.kind!=='reuse' && !live.designDraft ? runPage() : base.home()+`<div id="runtime-error" class="runtime-error data-error" role="alert">${esc(live.lastError)}</div>`;
  datasetCard = function() {
    const d=state.dataset;
    if(!d)return base.datasetCard();
    return `<div class="dataset-card"><div class="dataset-card-head"><div class="dataset-label">${icon('folder')}<div><strong>${esc(d.name)}</strong><small>${d.summary.sample_count} samples · ${d.summary.primary_image_count} primary images · ${d.summary.mask_count} masks</small></div></div><button class="subtle-link" data-action="pick-data">Change</button></div><p class="dataset-path">${esc(d.path)}</p>${d.demo?'<div class="dataset-previews"><div class="dataset-preview"><img src="assets/tau-wt.png" alt="WT_1 demo image"><span>WT_1</span></div><div class="dataset-preview"><img src="assets/tau-mu.png" alt="MU_1 demo image"><span>MU_1</span></div></div>':''}</div>`;
  };
  settings = () => base.settings().replace('Save changes',state.savingSettings?'Applying…':'Apply settings').replace('data-action="save-settings"',`data-action="save-settings" ${state.savingSettings?'disabled':''}`)+`<div id="runtime-error" class="runtime-error data-error" role="alert">${esc(live.lastError)}</div>`;
  analysisSettings = () => base.analysisSettings();
  apiSettings = () => base.apiSettings().replace('Enter your API key',state.config.hasApiKey?'Set for this session · leave blank to keep':'Enter your API key').replace('Enter your VLM API key',state.config.hasVlmApiKey?'Set for this session · leave blank to keep':'Enter your VLM API key');
  previewDoc=async function(id) {
    await task(async()=>{
      state.knowledge.error='';
      const preview=await get(`documents/${id}`);
      if(state.knowledge.open && state.docs.some(d=>d.id===id)){state.knowledge.preview=preview;render();}
    });
  };
  downloadDoc=async function(id) {
    await task(async()=>{
      const doc=state.docs.find(d=>d.id===id); if(!doc)return;
      const response=await api(`documents/${id}/file`);
      const url=URL.createObjectURL(await response.blob()), a=document.createElement('a');
      a.href=url;a.download=doc.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });
  };
  function remainingTime(j) {
    const seconds=j.eta?.remainingSeconds ?? (active(j)?j.initialEstimateSeconds:null);
    if(seconds==null)return 'Estimating…';
    if(seconds<60)return 'About 1 min';
    const minutes=Math.ceil(seconds/60);
    return 'About '+minutes+' min';
  }
  function savedResults(j) {
    if(j.exportStatus==='saving')return '<div class="automatic-results" role="status">'+icon('spinner','spinner')+'<div><strong>Saving results…</strong><p>Your timestamped folder and download will be ready shortly.</p></div></div>';
    if(j.exportStatus==='failed')return '<div class="automatic-results save-failed" role="alert"><div><strong>Results need to be packaged</strong><p>Raw results are preserved. '+esc(j.exportError||'Please retry saving.')+'</p></div>'+buttonHTML('Retry save','save-bundle')+'</div>';
    if(j.lastExport)return '<div class="automatic-results"><div>'+icon('check')+'<strong>Results saved'+(j.status==='partial'?' · partial run':'')+'</strong><p class="dataset-path">'+esc(j.lastExport.directory)+'</p></div><a class="button primary" data-action="download-bundle" href="/api/runs/'+encodeURIComponent(j.id)+'/export" download="'+esc(j.lastExport.filename)+'">'+icon('save')+'Download results (.zip)</a></div>';
    if(!active(j) && cards().length)return '<div class="automatic-results"><p>Package these imported results for download.</p>'+buttonHTML('Save result bundle','save-bundle')+'</div>';
    return '';
  }
  runPage = function() {
    const j=live.job;
    if(!j)return base.home();
    const isActive=active(j), complete=j.status==='complete';
    const stages=j.kind==='reuse'?[['quantify','Execute selected feature code'],['export','Save new measurements']]:[['inspect','Inspect data'],['prepare','Prepare images'],['plan','Plan features'],['quantify','Quantify features'],['validate','Validate and refine'],['export','Export results']];
    const current=stages.findIndex(s=>s[0]===j.stage);
    const action=isActive?buttonHTML('Stop run','live-stop'):(j.kind==='reuse'?buttonHTML('New computation','new-compute'):'')+(cards().length?buttonHTML('View results','view-features',true):'');
    const loops=j.kind==='reuse' ? `${j.completedMeasurements||0} / ${j.totalMeasurements||'—'} measurements` : j.currentRound ? 'Loop '+j.currentRound+' of '+j.rounds : j.rounds+' '+(j.rounds===1?'loop':'loops');
    const mode=Object.values(modes).find(m=>m.loops===j.rounds);
    return '<section class="page feature-design-run">'+
      '<div class="submitted-question"><span class="section-eyebrow">Your question</span><h1>'+esc(j.question||'Saved analysis')+'</h1></div>'+
      '<div class="design-run-header"><div class="design-run-title">'+statusIcon(j)+'<div><strong>'+statusLabel(j)+'</strong><small>'+displayName(j)+'</small></div></div>'+action+'</div>'+
      '<div class="run-meta"><span>'+icon('folder')+esc(j.datasetName||'Saved dataset')+'</span><span>'+icon('both')+esc(routes[j.route]?.name||j.route||'')+'</span><span>'+icon('clock')+'Elapsed '+elapsed()+'</span></div>'+
      (isActive?'<div class="eta-panel" role="status"><div><span>Estimated remaining</span><strong>'+remainingTime(j)+'</strong></div><span>'+esc(loops)+'</span></div>':'')+
      savedResults(j)+
      '<div class="run-status" role="status">'+esc(j.message||(isActive?'Running — logs stream below.':''))+'</div>'+
      '<div class="run-layout"><div>'+
      (j.status==='loaded'?'<p class="preview-note">Imported results are available for review. Completion of the original process has not been verified.</p>':
        '<div class="timeline">'+stages.map(([id,label],i)=>'<div class="timeline-item '+(complete||current>i?'done':isActive&&current===i?'current':'')+'"><span class="timeline-icon">'+icon(complete||current>i?'check':isActive&&current===i?'spinner':'clock',isActive&&current===i?'spinner':'')+'</span><div class="timeline-copy"><strong>'+label+'</strong></div></div>').join('')+'</div>')+
      '</div><aside class="run-summary"><h2>RUN DETAILS</h2>'+
      '<div class="summary-row"><small>'+ (j.kind==='reuse'?'Selected features':'Feature number')+'</small><strong>'+esc(j.kind==='reuse'?(j.selectedFeatures||[]).length:j.featureNumber??'—')+'</strong></div>'+
      '<div class="summary-row"><small>Analysis mode</small><strong>'+esc(j.kind==='reuse'?'Compute · saved code':(mode?mode.name+' · ':'')+loops)+'</strong></div>'+
      (j.kind==='reuse'?'':'<div class="summary-row"><small>Knowledge</small><strong>'+(j.referenceCount||0)+' uploaded files</strong></div>')+
      '<p class="preview-note">'+(j.kind==='reuse'?'Saved per-feature scripts only. No model calls; no new validation.':'Reproducibility enabled · seed 42. Your original inputs are preserved.')+'</p></aside></div>'+
      '<div id="runtime-error" class="runtime-error" role="alert">'+esc(live.lastError)+'</div>'+
      '<div class="live-log-header"><h2>Live output</h2><span>'+(isActive?'Process active':statusLabel(j))+'</span></div>'+
      '<pre class="live-log" id="live-log" aria-label="Run console">'+esc(live.logs.join('\n')||'Process output will appear here immediately after launch.')+'</pre>'+
      '<p class="dataset-path">Raw results: '+esc(j.resultsDir)+'</p></section>';
  };
  const computeFeatures = () => (live.reuseSource?.features||[]).filter(f=>f.reusable);
  featureRows = function() {
    return (live.reuseSource?.features||[]).filter(f=>`${f.name} ${f.category} ${f.method}`.toLowerCase().includes(state.search.toLowerCase())).map(f=>`<div class="compute-feature-row"><div><strong>${esc(f.name)}</strong><p>${esc(f.description||'No description saved.')}</p></div><span class="route-tag">${esc(f.method.toUpperCase())}</span><small>${f.reusable?'Ready to compute':'No executable code'}</small></div>`).join('')||empty('No matching features.');
  };
  reusePage = function() {
    if(live.job?.kind==='reuse' && !live.computeDraft)return runPage();
    const d=live.reuseData, source=live.reuseSource;
    return `<section class="home compute-home fade-in"><div class="home-heading">${mark('discovery-mark')}<h1>What would you like to compute?</h1><p>Apply your saved feature code to a new dataset.</p></div>
      <div class="composer"><textarea id="compute-question" aria-label="Compute question" placeholder="Describe what you want to measure on this dataset…">${esc(state.computeQuestion)}</textarea>
      <div class="composer-footer"><div class="composer-tools">
      <button class="composer-chip ${source?'filled':''}" data-action="compute-source">${icon('history')}<span>${source?'Previous run attached':'Upload features'}</span></button>
      <button class="composer-chip ${d?'filled':''}" data-action="reuse-upload">${icon(d?'folder':'plus')}<span>${d?'Data attached':'Add data'}</span></button>
      ${source?`<button class="composer-chip" data-action="compute-features" aria-expanded="${state.computeFeaturesOpen}">${icon('code')}<span>${computeFeatures().length} features available</span>${icon(state.computeFeaturesOpen?'down':'chevron')}</button>`:''}
      </div><button class="submit" data-action="prepare-compute" aria-label="Review and compute">${icon('arrow')}</button></div></div>
      <div class="compute-attachments">
      ${source?`<div class="compute-attachment">${icon('history')}<div><strong>${esc(source.name)}</strong><small>Previous run · ${computeFeatures().length} executable features available</small><p class="dataset-path">${esc(source.resultsDir)}</p></div><button class="subtle-link" data-action="compute-source">Change</button></div>`:''}
      ${d?`<div class="compute-attachment">${icon('folder')}<div><strong>${esc(d.name)}</strong><small>New dataset · ${d.summary.sample_count} samples · ${d.summary.primary_image_count} images</small><p class="dataset-path">${esc(d.path)}</p></div><button class="subtle-link" data-action="reuse-upload">Change</button></div>`:''}
      </div>
      ${source&&state.computeFeaturesOpen?`<section class="compute-feature-picker"><div class="compute-feature-heading"><h2>Saved features</h2></div><p class="muted-note">All available feature code will run automatically.</p><label class="search-box">${icon('search')}<input id="feature-search" type="search" placeholder="Search features…" aria-label="Search features" value="${esc(state.search)}"></label><div class="reuse-features" id="feature-list">${featureRows()}</div></section>`:''}
      ${!d?`<button class="starter compute-demo" data-action="reuse-demo"><span class="starter-copy"><strong>Try with the Tau demo</strong><span>Use the demo as your new target dataset</span></span>${icon('right')}</button>`:''}
      <div id="runtime-error" class="runtime-error" role="alert">${esc(live.lastError)}</div></section>`;
  };
  const featureBadge = f => `<span class="feature-status ${f.method==='vlm'?'vlm':'code'}">${f.method==='vlm'?'VLM':'Code'}</span>`;
  function histogram(data,feature) {
    if(!data.count || !data.bins.length)return empty('No distribution data saved for this feature.');
    const bins=data.bins, left=62, top=30, width=560, height=280;
    const max=Math.max(...bins.map(b=>b.count)), step=Math.max(1,Math.ceil(max/4)), ceiling=step*4;
    const format=n=>Number(n.toPrecision(3)).toString();
    const low=bins[0].low, high=bins[bins.length-1].high;
    const grid=Array.from({length:5},(_,i)=>{
      const y=top+height-i*height/4;
      return `<line x1="${left}" x2="${left+width}" y1="${y}" y2="${y}" class="histogram-grid"/><text x="${left-12}" y="${y+4}" text-anchor="end">${i*step}</text>`;
    }).join('');
    const ticks=Array.from({length:5},(_,i)=>`<text x="${left+width*i/4}" y="${top+height+25}" text-anchor="middle">${esc(format(low*(1-i/4)+high*i/4))}</text>`).join('');
    return `<svg class="feature-histogram" viewBox="0 0 650 390" role="img" aria-labelledby="histogram-title histogram-description"><title id="histogram-title">${esc(feature.name)} distribution</title><desc id="histogram-description">Histogram of ${data.count} samples. Horizontal axis: feature value. Vertical axis: sample count.</desc>${grid}${bins.map((b,i)=>{
      const h=b.count/ceiling*height;
      return `<rect class="histogram-bar" x="${left+i*width/bins.length+1}" y="${top+height-h}" width="${width/bins.length-2}" height="${h}" rx="2"><title>${esc(format(b.low))} – ${esc(format(b.high))}: ${b.count} samples</title></rect>`;
    }).join('')}${ticks}<text x="${left+width/2}" y="377" text-anchor="middle" class="histogram-axis">Feature value</text><text transform="translate(17 170) rotate(-90)" text-anchor="middle" class="histogram-axis">Samples</text></svg><p class="histogram-count">${data.count} samples${data.missing?` · ${data.missing} missing / non-finite excluded`:''}</p>`;
  }
  visualizePage = function() {
    const items=visibleFeatures(), f=items.find(item=>item.index===state.vizFeature)?.feature, job=live.visualizeJob;
    if(!job)return `<section class="page visualize-empty fade-in"><div class="visualize-upload-icon">${icon('chart')}</div><h1>Visualize</h1>${buttonHTML('Upload feature_value.csv','visualize-upload',true)}</section>`;
    const filters=`<div class="distribution-filters" role="group" aria-label="Filter features by extraction method">${[['all','All'],['code','Code'],['vlm','VLM']].map(([value,label])=>`<button class="distribution-filter" data-action="visualize-filter" data-value="${value}" aria-pressed="${live.visualizeFilter===value}"><span class="filter-check" aria-hidden="true">${icon('check')}</span>${label}</button>`).join('')}</div>`;
    return `<section class="page visualize-page fade-in">${pageHeading('Visualize',esc(job.name),buttonHTML('Upload feature_value.csv','visualize-upload'))}<div class="distribution-layout"><div class="distribution-sidebar">${filters}<div class="distribution-selector" aria-label="Features">${items.map(({feature:f,index:i})=>`<button class="viz-feature ${i===state.vizFeature?'active':''}" data-action="live-viz" data-index="${i}" aria-pressed="${i===state.vizFeature}"><strong>${esc(f.name)}</strong>${featureBadge(f)}</button>`).join('')||empty(live.visualizeFilter==='all'?'No features saved in this run.':`No ${live.visualizeFilter.toUpperCase()} features.`)}</div></div><div class="distribution-main">${f?`<div class="distribution-heading"><h2>${esc(f.name)}</h2>${featureBadge(f)}</div>${live.distribution?histogram(live.distribution,f):live.distributionError?`<div class="distribution-error" role="alert">${esc(live.distributionError)}${buttonHTML('Retry','live-viz',false,`data-index="${state.vizFeature}"`)}</div>`:`<div class="distribution-loading" role="status">${icon('spinner','spinner')}Loading distribution…</div>`}`:''}</div></div></section>`;
  };

  function newRun() {live.designDraft=true;live.lastError='';state.question='';state.featureNumber='';state.runDialog=null;state.dataset=null;state.historyOpen=false;navigate('data');}
  async function reviewHistoryRemoval(id) {
    const job=live.runs.find(j=>j.id===id);if(!job)return;
    if(active(job)||job.exportStatus==='saving')return showRunIssue({title:'Run is still active',message:'Wait until this run and result saving finish before removing it from History.'});
    const plan=await get(`runs/${id}/removal`);
    state.runDialog={kind:'delete-history',id,name:job.name,paths:plan.paths,preservedPaths:plan.preservedPaths,launching:false};render();
  }
  async function confirmHistoryRemoval() {
    const d=state.runDialog;if(d?.kind!=='delete-history'||d.launching)return;
    d.launching=true;render();
    try {
      const result=await post(`runs/${d.id}/remove`);
      live.runs=live.runs.filter(j=>j.id!==d.id);
      if(live.reuseSource?.id===d.id)live.reuseSource=null;
      if(live.visualizeJob?.id===d.id){live.visualizeJob=null;live.distribution=null;live.distributionError='';}
      if(live.job?.id===d.id){
        live.job=null;live.logs=[];live.offset=0;live.evidence=null;live.preview=null;
        live.objectUrls.forEach(URL.revokeObjectURL);live.objectUrls=[];
        live.designDraft=true;live.computeDraft=true;state.historyOpen=false;
      }
      state.runDialog=null;render();toast(result.systemTrash?'Run deleted. Files moved to system Trash / Recycle Bin.':'History entry deleted. External or shared files were kept.');
    } catch(e) {showRunIssue({title:'Could not remove this record',message:e.message,label:'Back to History'});}
  }
  async function reviewAnalysis() {
    if(state.knowledge.uploading || state.runDialog)return;
    const issue=submissionIssue();if(issue)return showRunIssue(issue);
    try {
      setSettings(await post('settings',state.config));
      const connectionIssue=submissionIssue();if(connectionIssue)return showRunIssue(connectionIssue);
      const request=payload();
      const check=await post('preflight',request);
      if(!check.ready)return showRunIssue({title:'Cannot start this run',message:check.issues.filter(i=>i.severity==='blocker').map(i=>i.message).join('\n'),label:'Review inputs'});
      if(JSON.stringify(request)!==JSON.stringify(payload()))return showRunIssue({title:'Inputs changed',message:'Your inputs changed during the checks. Submit again to review the updated configuration.',label:'Review inputs'});
      showRunConfirmation(request);
    } catch(e) {
      showRunIssue({title:'Configuration needs attention',message:e.message,label:'Review inputs'});
    }
  }
  async function launchConfirmedAnalysis() {
    const dialog=state.runDialog;
    if(dialog?.kind!=='confirm' || dialog.launching)return;
    dialog.launching=true;render();
    try {
      const compute=dialog.workflow==='compute';
      const j=await post(compute?'compute':'runs',dialog.request);
      // Only consume the draft that was submitted; the other workflow stays intact.
      if(compute){live.computeDraft=false;state.computeQuestion='';live.reuseSource=null;live.reuseData=null;}
      else {live.designDraft=false;state.question='';state.featureNumber='';state.dataset=null;}
      // Clear the reviewed request immediately: retrying a click cannot start it twice.
      state.runDialog=null;live.job=j;live.logs=[];render();window.scrollTo(0,0);
      await refreshHistory();await loadJob(j.id);toast('Analysis started.');
    } catch(e) {
      if(state.runDialog===dialog)showRunIssue({title:'Unable to start analysis',message:e.message,label:'Back to configuration'});
      else error(e);
    }
  }
  prepareRun = () => task(reviewAnalysis);
  function computePayload() {return {sourceRunId:live.reuseSource?.id,datasetId:live.reuseData?.id,featureNames:computeFeatures().map(f=>f.name),question:state.computeQuestion.trim()};}
  async function reviewCompute() {
    if(state.runDialog)return;
    if(!live.reuseSource)return showRunIssue({title:'Feature folder required',message:'Upload the feature folder from a saved run.',label:'Choose feature folder'});
    if(!live.reuseData)return showRunIssue({title:'Dataset required',message:'Add a new target dataset before continuing.',label:'Add data'});
    if(!state.computeQuestion.trim())return showRunIssue({title:'Question required',message:'Describe what you want to measure on this dataset.',target:'compute-question',label:'Enter your question'});
    if(!computeFeatures().length)return showRunIssue({title:'No executable features',message:'This run has no standalone feature code. Load another previous run.',label:'Review previous run'});
    const issue=apiConnectionIssue('code');if(issue)return showRunIssue(issue);
    try {
      setSettings(await post('settings',state.config));
      const issue=apiConnectionIssue('code');if(issue)return showRunIssue(issue);
      const request=computePayload();
      const check=await post('compute/preflight',request);
      if(!check.ready)return showRunIssue({title:'Cannot start computation',message:check.issues.map(i=>i.message).join('\n'),label:'Review inputs'});
      if(JSON.stringify(request)!==JSON.stringify(computePayload()))return showRunIssue({title:'Inputs changed',message:'Submit again to review your updated inputs.'});
      const d=live.reuseData, source=live.reuseSource;
      state.runDialog={kind:'confirm',workflow:'compute',launching:false,request:JSON.parse(JSON.stringify(request)),summary:{
        question:request.question,source:{name:source.name,path:source.resultsDir},features:[...request.featureNames],
        dataset:{name:d.name,path:d.path,samples:d.summary.sample_count,images:d.summary.primary_image_count}
      }};render();
    } catch(e) {showRunIssue({title:'Configuration needs attention',message:e.message,label:'Review inputs'});}
  }
  async function selectReuseSource(id) {live.reuseSource=await get('runs/'+id);live.computeDraft=true;state.historyOpen=false;state.search='';state.computeFeaturesOpen=false;navigate('compute');}
  async function importDataset(files) {
    const valid=files.filter(f=>/\.(tiff?|png|jpe?g|mrc|csv|json|txt|md)$/i.test(f.name)&&!(f.webkitRelativePath||'').split('/').some(p=>p.startsWith('.')||p==='results'));
    if(!valid.length)throw new Error('No supported microscopy files found.');
    const upload=await post('imports');
    for(let i=0;i<valid.length;i++){const f=valid[i];toast(`Importing ${i+1}/${valid.length}: ${f.name}`);await api(`imports/${upload.id}?name=${encodeURIComponent(f.webkitRelativePath||f.name)}`,{method:'PUT',body:f});}
    return post(`imports/${upload.id}/finish`);
  }
  const intercepted=new Set(['new','live-new','demo','history','live-library','live-example','load-job','load-path','save-settings','prepare-run','confirm-run','live-stop','stop-preview','view-features','live-feature','live-viz','preview-artifact','download-artifact','live-reuse','navigate','remove-doc','reuse-source','reuse-upload','reuse-path','reuse-demo','save-bundle','download-bundle']);
  ['compute-source','compute-features','prepare-compute','new-compute','delete-history','confirm-delete-history','visualize-upload','visualize-filter','help-send','help-clear'].forEach(a=>intercepted.add(a));
  app.addEventListener('click',event=>{
    const b=event.target.closest('[data-action]');if(!b||b.disabled||!intercepted.has(b.dataset.action))return;
    event.stopImmediatePropagation();event.preventDefault();
    const action=b.dataset.action;
    if(action==='help-send'){sendHelp();return;}
    task(async()=>{
      if(action==='new'||action==='live-new')newRun();
      else if(action==='demo')await chooseDemo();
      else if(action==='save-settings')await saveSettings();
      else if(action==='help-clear'){state.help={draft:'',messages:[],pending:false,error:''};render();}
      else if(action==='remove-doc'){
        const id=b.dataset.id; if(!state.docs.some(d=>d.id===id))return;
        await post(`documents/${id}/remove`);state.docs=state.docs.filter(d=>d.id!==id);
        state.knowledge.error='';if(state.knowledge.preview?.id===id)state.knowledge.preview=null;
        render();toast('Reference removed from future runs.');
      }
      else if(action==='live-example'||action==='history'){const j=await post('runs/demo');await refreshHistory();await loadJob(j.id,'visualize');}
      else if(action==='load-job')await loadJob(b.dataset.id);
      else if(action==='delete-history')await reviewHistoryRemoval(b.dataset.id);
      else if(action==='confirm-delete-history')await confirmHistoryRemoval();
      else if(action==='load-path'){const path=window.prompt('Paste the local results folder path (contains features.csv):');if(path){const j=await post('runs/load',{path});await refreshHistory();if(state.page==='compute')await selectReuseSource(j.id);else await loadJob(j.id,'visualize');}}
      else if(action==='live-library'||action==='new-compute'){live.computeDraft=true;state.historyOpen=false;state.computeQuestion='';navigate('compute');}
      else if(action==='compute-source'){
        const path=window.prompt('Choose the feature folder from a saved run:',live.exportsDirectory);
        if(path){try{const j=await post('runs/load',{path});await refreshHistory();await selectReuseSource(j.id);}catch(e){showRunIssue({title:'Unable to load previous run',message:e.message,label:'Choose another folder'});}}
      }
      else if(action==='visualize-upload'){
        const path=window.prompt('Choose feature_value.csv to visualize:',live.exportsDirectory);
        if(path){try{const j=await post('runs/load',{path});const job=await get('runs/'+j.id);await refreshHistory();await showVisualization(job);}catch(e){showRunIssue({title:'Unable to load run',message:e.message,label:'Choose another folder'});}}
      }
      else if(action==='compute-features'){state.computeFeaturesOpen=!state.computeFeaturesOpen;render();}
      else if(action==='prepare-compute'||action==='live-reuse')await reviewCompute();
      else if(action==='view-features')await showVisualization(live.job);
      else if(action==='navigate')await openWorkflow(b.dataset.page);
      else if(action==='live-feature')await showVisualization(live.job,cards().findIndex(c=>c.name===b.dataset.name));
      else if(action==='live-viz')await loadDistribution(Number(b.dataset.index));
      else if(action==='visualize-filter')await filterVisualization(b.dataset.value);
      else if(action==='preview-artifact')await previewArtifact(b.dataset.path,b.dataset.type);
      else if(action==='download-artifact')await downloadArtifact(b.dataset.path);
      else if(action==='reuse-source')await selectReuseSource(b.dataset.id);
      else if(action==='reuse-upload')document.getElementById('reuse-dataset-input').click();
      else if(action==='reuse-demo'){live.reuseData=await post('datasets/demo');render();}
      else if(action==='reuse-path'){const path=window.prompt('Paste the new dataset folder path (contains dataset/):');if(path){live.reuseData=await post('datasets',{path});render();}}
      else if(action==='save-bundle'){b.disabled=true;b.textContent='Saving…';const id=live.job.id;await post(`runs/${id}/export`);if(live.job?.id===id)live.job=await get('runs/'+id);render();toast('Result bundle saved.');}
      else if(action==='download-bundle'){
        const response=await api(`runs/${live.job.id}/export`);
        const url=URL.createObjectURL(await response.blob());const a=document.createElement('a');a.href=url;a.download=live.job.lastExport.filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      }
      else if(action==='live-stop'||action==='stop-preview'){if(live.job){await post(`runs/${live.job.id}/cancel`);live.job=await get('runs/'+live.job.id);render();}}
      else if(action==='prepare-run')await reviewAnalysis();
      else if(action==='confirm-run')await launchConfirmedAnalysis();
    });
  },true);
  app.addEventListener('input',event=>{if(event.target.id==='compute-question')state.computeQuestion=event.target.value;});
  app.addEventListener('keydown',event=>{
    if(event.target.id==='compute-question' && event.key==='Enter' && !event.shiftKey && !event.isComposing){event.preventDefault();task(reviewCompute);}
  });
  addDocs=async function(files) {
    const selected=[...files]; if(!selected.length)return;
    await task(async()=>{
      state.knowledge.open=true;state.knowledge.preview=null;state.knowledge.error='';state.knowledge.uploading=true;
      const failures=[];let added=0;
      try {
        for(const file of selected){
          state.knowledge.progress='Preparing '+file.name+'…';render();
          try {
            const item=await (await api('documents?name='+encodeURIComponent(file.name),{method:'POST',body:file})).json();
            state.docs.push(item);added++;
          } catch(e) {failures.push(file.name+': '+e.message);}
        }
      } finally {
        state.knowledge.uploading=false;state.knowledge.progress='';state.knowledge.error=failures.join('\n');render();
      }
      toast(failures.length ? `${added} files attached. Check the upload errors.` : 'Knowledge attached. Ready for your next run.');
    });
  };
  document.getElementById('dataset-input').addEventListener('change',event=>{
    event.stopImmediatePropagation();const files=[...event.target.files];event.target.value='';if(!files.length)return;
    task(async()=>{
      state.dataset=await importDataset(files);render();toast(`${state.dataset.summary.sample_count} samples imported. Submit your question to start.`);
    });
  },true);
  document.getElementById('reuse-dataset-input').addEventListener('change',event=>{
    const files=[...event.target.files];event.target.value='';if(!files.length)return;
    task(async()=>{live.reuseData=await importDataset(files);render();toast('Target dataset ready. Review your inputs to compute.');});
  });
  document.addEventListener('keydown',e=>{
    if(state.runDialog || state.knowledge.open)return;
    if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.stopImmediatePropagation();e.preventDefault();newRun();}
  },true);
  async function poll() {
    try {
      if(live.job&&(active(live.job)||live.job.exportStatus==='saving')){
        const key=live.job.id, data=await get('runs/'+key);
        if(live.job?.id===key){
          live.job=data;const old=live.runs.find(r=>r.id===key);if(old)Object.assign(old,data);
          await updateLogs();
          if(live.job?.id===key && showingRun()){
            const l=document.getElementById('live-log');
            const atBottom=!l||l.scrollHeight-l.scrollTop-l.clientHeight<60;
            render();
            if(atBottom){const log=document.getElementById('live-log');if(log)log.scrollTop=log.scrollHeight;}
          }
          if(!active(data))await refreshHistory();
        }
      }
      else if(live.runs.some(active)){await refreshHistory();}
      live.connected=true;
    } catch(e){live.connected=false;toast('Local service disconnected. Keep its terminal running, then reload.');}
    setTimeout(poll,1500);
  }
  refreshHistory().then(data=>{
    setSettings(data.settings);state.docs=data.documents;features.splice(0,features.length);
    document.title='MorphAgent · Workspace';
    const running=live.runs.find(active);
    if(running)loadJob(running.id).catch(error);else render();poll();
  }).catch(e=>{app.innerHTML='<div class="empty">Could not connect to the local runtime. Reload this page after restarting launch_web_ui.py.</div>';error(e);});
})();
