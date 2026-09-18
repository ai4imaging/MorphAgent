/* Standalone design prototype. No model calls, analytics, or backend requests. */
const icons = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  new: '<path d="M12 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-6"/><path d="m16 3 5 5-10 10H6v-5Z"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/>',
  settings: '<path d="m10 3-.6 2.1-1.8 1L5.4 6 3.5 9.3l1.6 1.6v2.2l-1.6 1.6L5.4 18l2.2-.1 1.8 1L10 21h4l.6-2.1 1.8-1 2.2.1 1.9-3.3-1.6-1.6v-2.2l1.6-1.6L18.6 6l-2.2.1-1.8-1L14 3Z"/><circle cx="12" cy="12" r="3"/>',
  folder: '<path d="M3 8V6a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
  run: '<path d="m8 4 13 8L8 20Z"/>',
  reuse: '<path d="M20 8a8 8 0 0 0-14-3L3 8m0-5v5h5M4 16a8 8 0 0 0 14 3l3-3m0 5v-5h-5"/>',
  chart: '<path d="M4 4v16h16M8 15l4-5 4 3 4-7"/>',
  save: '<path d="M12 3v12m-4-4 4 4 4-4M4 15v5h16v-5"/>',
  arrow: '<path d="M12 19V5m-6 6 6-6 6 6"/>',
  right: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  left: '<path d="M20 12H4m6-6-6 6 6 6"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
  down: '<path d="m8 10 4 4 4-4"/>',
  lightning: '<path d="m13 2-9 12h7l-1 8 10-13h-8Z"/>',
  fast: '<path d="m4 6 6 6-6 6m8-12 6 6-6 6"/>',
  detailed: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><path d="M12 1v3m0 16v3M1 12h3m16 0h3"/>',
  code: '<path d="m7 6-6 6 6 6m10-12 6 6-6 6m-4-15-2 18"/>',
  vision: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m4 18 6-6 4 4 3-3 4 5"/>',
  both: '<rect x="3" y="3" width="8" height="8" rx="2"/><rect x="13" y="13" width="8" height="8" rx="2"/><path d="M15 3h6v6M3 15v6h6"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h5"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0v4"/>',
  eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
  book: '<path d="M12 6S7 2 2 5v15c5-3 10 1 10 1s5-4 10-1V5c-5-3-10 1-10 1Zm0 0v15"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  spinner: '<path d="M21 12a9 9 0 1 1-9-9"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  history: '<path d="M3 10a9 9 0 1 1 1 7M3 4v6h6M12 7v5l3 2"/>',
  spark: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z"/>',
  help: '<path d="M4 5h16v13H8l-4 3V5Z"/><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.5-2.5 2-2.5 3"/><path d="M12 15h.01"/>',
};
const icon = (name, cls = '') => `<svg class="icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.file}</svg>`;
const mark = (cls) => `<svg class="${cls}" viewBox="0 0 40 40" fill="none" aria-hidden="true"><path d="M7 28V12l8 13 5-9 5 9 8-13v16" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/><path d="M10 7c6-4 14-4 20 0M10 33c6 4 14 4 20 0" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;
const escapeHTML = (s = '') => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const modes = {
  ultra: { name: 'Ultra fast', loops: 1, icon: 'lightning', description: 'A quick first look' },
  fast: { name: 'Fast', loops: 5, icon: 'fast', description: 'Explore and refine' },
  detailed: { name: 'Detailed', loops: 20, icon: 'detailed', description: 'A deeper investigation' },
};
const routes = {
  both: { name: 'Code + VLM', icon: 'both', description: 'Measurements and visual reasoning' },
  code: { name: 'Code only', icon: 'code', description: 'Code-based measurements' },
  vlm: { name: 'VLM only', icon: 'vision', description: 'Visual-language assessments' },
};
const features = [
  { id: 'axon_to_soma_tau_intensity_ratio', name: 'Axon-to-soma Tau intensity ratio', route: 'Code', category: 'Distribution', description: 'Compares mean Tau signal in axonal and somatic regions to describe its subcellular distribution.' },
  { id: 'tau_neurite_bead_density', name: 'Tau neurite bead density', route: 'Code', category: 'Morphology', description: 'Describes the density of bead-like Tau structures along neuronal processes.' },
  { id: 'vlm_tau_aggregation_pattern_proportion', name: 'Tau aggregation pattern proportion', route: 'VLM', category: 'Morphology', description: 'Assesses the relative extent of visible Tau aggregation patterns in an image.' },
  { id: 'dendritic_tau_thread_density', name: 'Dendritic Tau thread density', route: 'Code', category: 'Morphology', description: 'Describes the density of thread-like Tau structures in dendritic regions.' },
  { id: 'vlm_subcellular_tau_distribution_ratio', name: 'Subcellular Tau distribution', route: 'VLM', category: 'Distribution', description: 'Summarizes the visible distribution of Tau across neuronal compartments.' },
  { id: 'soma_nft_compactness', name: 'Somatic aggregate compactness', route: 'Code', category: 'Morphology', description: 'Characterizes the spatial compactness of segmented somatic Tau aggregates.' },
];
const state = {
  page: 'data', previousPage: 'data', settingsTab: 'api', sidebarOpen: false,
  question: '', featureNumber: '', dataset: null, historyOpen: false, runDialog: null,
  computeFeaturesOpen: false,
  config: { route: 'both', mode: 'ultra', baseUrl: '', apiKey: '', model: '', sameConnection: true, vlmBaseUrl: '', vlmApiKey: '', vlmModel: '', knowledgeEnabled: true },
  docs: [], selectedFeatures: new Set(), search: '', vizFeature: 0, sample: 'wt',
  knowledge: {open:false, uploading:false, error:'', progress:'', preview:null},
  help: {draft:'', messages:[], pending:false, error:''},
  runStatus: 'idle', runStep: 0, timer: null, editingSecrets: new Set(), previewSubmitted: false,
};
const app = document.getElementById('app');
function field(label, key, placeholder, secret = false) {
  const hasKey=state.config[key==='apiKey'?'hasApiKey':'hasVlmApiKey'];
  const retained=secret && hasKey && !state.config[key] && !state.editingSecrets.has(key);
  // A display-only mask: the real saved key never comes back from the service.
  const value=retained?'********':state.config[key];
  return `<label class="form-label">${label}<span class="input-wrap"><input data-config="${key}" type="${secret && !retained ? 'password' : 'text'}" value="${escapeHTML(value)}" placeholder="${placeholder}" autocomplete="off" spellcheck="false" ${secret ? 'data-secret="true"' : ''} ${state.savingSettings?'disabled':''} ${retained?'readonly data-retained-secret="true" aria-label="API key configured for this session"':''} />${retained?`<button class="subtle-link secret-change" type="button" data-action="edit-api-key" data-key="${key}" ${state.savingSettings?'disabled':''}>Change</button>`:secret?`<button class="icon-button" type="button" data-action="toggle-secret" data-key="${key}" aria-label="Show API key">${icon('eye')}</button>`:''}</span></label>`;
}
const toggle = (key, label) => `<button class="switch" data-action="toggle" data-key="${key}" role="switch" aria-label="${label}" aria-checked="${state.config[key]}"><span></span></button>`;
const button = (text, action, symbol = '', primary = false, extra = '') => `<button class="button ${primary ? 'primary' : ''}" data-action="${action}" ${extra}>${symbol ? icon(symbol) : ''}${text}</button>`;
const pageHeading = (title, desc, action = '') => `<div class="page-heading"><div><h1>${title}</h1>${desc ? `<p>${desc}</p>` : ''}</div>${action}</div>`;

function workflowNavigation() {
  return '<nav class="sidebar-actions" aria-label="Analysis workflow">'+[['data','new','Design'],['compute','reuse','Compute'],['visualize','chart','Visualize'],['help','help','Help']].map(([id,symbol,label])=>`<button class="nav-action ${state.page===id?'active':''}" data-action="navigate" data-page="${id}" ${state.page===id?'aria-current="page"':''}>${icon(symbol)}<span>${label}</span></button>`).join('')+'</nav>';
}
function sidebar() {
  return `<aside class="sidebar ${state.sidebarOpen ? 'open' : ''}" aria-label="Workspace sidebar">
    <div class="brand">${mark('brand-mark')}<span>MorphAgent</span><button class="icon-button sidebar-close" data-action="sidebar" aria-label="Close sidebar">${icon('close')}</button></div>
    ${workflowNavigation()}
    <div class="history"><div class="section-eyebrow"><span>Example run</span>${icon('history', 'small-icon')}</div><button class="history-item ${state.historyOpen ? 'active' : ''}" data-action="history"><span>Tau morphology study</span><small>Demo · explore the workspace</small></button></div>
    <div class="sidebar-bottom"><button class="nav-action ${state.page === 'settings' ? 'active' : ''}" data-action="settings">${icon('settings')}<span>Settings</span></button></div>
  </aside>`;
}
function header() {
  return `<button class="mobile-menu" data-action="sidebar" aria-label="Open sidebar">${icon('menu')}</button>`;
}
function datasetCard() {
  const d = state.dataset;
  if (!d) return `<button class="starter" data-action="demo"><span class="sample-thumbs"><img src="assets/tau-wt.png" alt="Tau microscopy sample" /><img src="assets/tau-mu.png" alt="Second Tau microscopy sample" /></span><span class="starter-copy"><strong>Take a look with the Tau demo</strong><span>A ready-to-explore microscopy dataset</span></span>${icon('right')}</button>`;
  return `<div class="dataset-card"><div class="dataset-card-head"><div class="dataset-label">${icon('folder')}<div><strong>${escapeHTML(d.name)}</strong><small>${d.demo ? 'Demo dataset · 2 sample previews' : `${d.count} image file${d.count === 1 ? '' : 's'} · selected locally`}</small></div></div><button class="subtle-link" data-action="pick-data">Change</button></div>${d.demo ? `<div class="dataset-previews"><div class="dataset-preview"><img src="assets/tau-wt.png" alt="WT_1 sample" /><span>WT_1</span></div><div class="dataset-preview"><img src="assets/tau-mu.png" alt="MU_1 sample" /><span>MU_1</span></div></div>` : ''}</div>`;
}
function home() {
  if(state.previewSubmitted)return runPage();
  const mode = modes[state.config.mode];
  return `<section class="home fade-in"><div class="home-heading">${mark('discovery-mark')}<h1>What would you like to discover?</h1><p>Turn microscopy images into biologically grounded features.</p></div>
    <div class="composer"><textarea id="question" aria-label="Biological question" placeholder="Describe your biological question…">${escapeHTML(state.question)}</textarea>
      <div class="composer-footer"><div class="composer-tools">
        <button class="composer-chip ${state.dataset ? 'filled' : ''}" data-action="pick-data">${icon(state.dataset ? 'folder' : 'plus')}<span>${state.dataset ? 'Data attached' : 'Add data'}</span></button>
        ${knowledgeChip()}<span class="composer-divider"></span>
        <button class="composer-chip route-chip" data-action="settings" data-tab="analysis">${icon(routes[state.config.route].icon)}${routes[state.config.route].name}</button>
        <button class="composer-chip" data-action="settings" data-tab="analysis">${icon(mode.icon)}${mode.name}<span>· ${mode.loops} ${mode.loops === 1 ? 'loop' : 'loops'}</span>${icon('down')}</button>
        <label class="feature-count-field" for="feature-number"><span>Feature number <span class="required-mark" aria-hidden="true">*</span></span><input id="feature-number" type="number" min="1" max="500" step="1" inputmode="numeric" required aria-label="Feature number" placeholder="Required" title="Target feature count · whole number from 1 to 500" value="${escapeHTML(state.featureNumber)}"></label>
      </div><button class="submit" data-action="prepare-run" aria-label="Review run configuration" title="Review configuration" ${state.knowledge.uploading ? 'disabled' : ''}>${icon('arrow')}</button></div>
    </div>${datasetCard()}</section>`;
}
function helpMarkdown(source) {
  const lines=String(source||'').split(/\r?\n/), output=[];
  let list='';
  const inline=text=>escapeHTML(text).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  const closeList=()=>{if(list){output.push(`</${list}>`);list='';}};
  for(const line of lines){
    if(!line.trim()){closeList();continue;}
    const heading=line.match(/^(#{1,3})\s+(.+)$/);
    if(heading){closeList();output.push(`<h${heading[1].length+2}>${inline(heading[2])}</h${heading[1].length+2}>`);continue;}
    const bullet=line.match(/^\s*[-*]\s+(.+)$/),numbered=line.match(/^\s*\d+\.\s+(.+)$/);
    if(bullet||numbered){const next=bullet?'ul':'ol';if(list!==next){closeList();output.push(`<${next}>`);list=next;}output.push(`<li>${inline((bullet||numbered)[1])}</li>`);continue;}
    closeList();output.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return output.join('');
}
function helpPage() {
  const h=state.help, started=h.messages.length||h.pending;
  const greeting='I’m MorphAgent. I can help answer your questions about this paper, its methods, results, figures, supplementary material, and implementation.';
  const messages=[{role:'assistant',content:greeting},...h.messages];
  const composer=`<div class="help-composer composer"><textarea id="help-question" aria-label="Ask a question about the paper" placeholder="Ask about the paper, methods, figures, or code…" ${h.pending?'disabled':''}>${escapeHTML(h.draft)}</textarea><div class="composer-footer"><span class="help-composer-note">Answers use the maintained manuscript and code knowledge.</span><button class="submit" data-action="help-send" aria-label="Send question" ${h.pending?'disabled':''}>${icon('arrow')}</button></div></div>`;
  return `<section class="help-page ${started?'started':'empty'} fade-in">${started?`<header class="help-header"><h1>Help</h1><button class="button" data-action="help-clear" ${h.pending?'disabled':''}>New conversation</button></header><div class="help-thread" aria-live="polite">${messages.map(message=>`<div class="help-message ${message.role}"><span class="help-avatar">${message.role==='assistant'?mark('help-mark'):'You'}</span><div class="help-bubble"><div class="help-role">${message.role==='assistant'?'MorphAgent':'You'}</div><div class="help-markdown">${helpMarkdown(message.content)}</div></div></div>`).join('')}${h.pending?`<div class="help-message assistant"><span class="help-avatar">${mark('help-mark')}</span><div class="help-bubble help-thinking" role="status"><div class="help-role">MorphAgent</div><span>Reading the paper and preparing an answer</span><span class="help-dots"><i></i><i></i><i></i></span></div></div>`:''}</div>`:`<div class="home-heading">${mark('discovery-mark')}<h1>Ask MorphAgent</h1><p>Questions about the paper, its evidence, and the implementation.</p></div><div class="help-intro">${helpMarkdown(greeting)}</div>`}${h.error?`<p class="help-error" role="alert">${escapeHTML(h.error)}</p>`:''}${composer}</section>`;
}
function sendHelp() { toast('Open the desktop workspace to ask MorphAgent. This is an offline preview.'); }
function settings() {
  return `<section class="page settings-page fade-in">${pageHeading('Settings','',`<button class="button text" data-action="back">${icon('left')}Back to workspace</button>`)}<nav class="settings-tabs" aria-label="Settings sections">${[['api','Model API'],['analysis','Analysis']].map(([id,name]) => `<button class="settings-tab ${state.settingsTab === id ? 'active' : ''}" data-action="settings-tab" data-tab="${id}" ${state.settingsTab === id ? 'aria-current="page"' : ''}>${name}</button>`).join('')}</nav><div class="settings-body">${state.settingsTab === 'analysis' ? analysisSettings() : apiSettings()}</div><div class="settings-footer"><div>${button('Save changes','save-settings','',true)}</div></div></section>`;
}
function apiSettings() {
  return `<section class="setting-section"><div class="form-grid">${field('Base URL','baseUrl','https://api.your-provider.com/v1')}${field('API key','apiKey','Enter your API key',true)}${field('Model','model','Enter a model name')}</div><div class="connection-row"><div class="row-copy">Use the same connection for VLM</div>${toggle('sameConnection','Use the same connection for VLM')}</div>${!state.config.sameConnection ? `<div class="separate-vlm"><h2>VLM connection</h2><div class="form-grid">${field('Base URL','vlmBaseUrl','https://api.your-provider.com/v1')}${field('API key','vlmApiKey','Enter your VLM API key',true)}${field('Vision model','vlmModel','Enter a vision-capable model name')}</div></div>` : ''}</section>`;
}
function analysisSettings() {
  return `<section class="setting-section"><h2>Analysis route</h2><div class="choice-grid" role="group" aria-label="Analysis route">${Object.entries(routes).map(([key,r]) => `<button class="choice ${state.config.route === key ? 'selected' : ''}" data-action="route" data-value="${key}" aria-pressed="${state.config.route === key}"><span class="choice-top">${icon(r.icon)}<span class="radio-mark"></span></span><strong>${r.name}</strong><small>${r.description}</small></button>`).join('')}</div></section><section class="setting-section"><h2>Discovery depth</h2><div class="choice-grid" role="group" aria-label="Discovery depth">${Object.entries(modes).map(([key,m]) => `<button class="choice ${state.config.mode === key ? 'selected' : ''}" data-action="mode" data-value="${key}" aria-pressed="${state.config.mode === key}"><span class="choice-top">${icon(m.icon)}<span class="radio-mark"></span></span><strong>${m.name}</strong><small>${m.description}</small><span class="loop-count">${m.loops} ${m.loops === 1 ? 'loop' : 'loops'}</span></button>`).join('')}</div></section>`;
}
function knowledgeChip() {
  const count = state.docs.length;
  return `<button class="composer-chip knowledge-chip ${count ? 'filled' : ''}" data-action="manage-knowledge" aria-haspopup="dialog" title="${count ? 'View, add or remove knowledge files' : 'Attach papers or expert notes'}">${icon(state.knowledge.uploading ? 'spinner' : 'book',state.knowledge.uploading ? 'spinner' : '')}<span>${count ? 'Knowledge attached' : 'Upload knowledge'}</span>${count ? `<span class="attachment-count" aria-label="${count} files">${count}</span>` : ''}</button>`;
}
function knowledgeDialog() {
  if (!state.knowledge.open) return '';
  const k=state.knowledge, preview=k.preview;
  const list=state.docs.length ? `<div class="knowledge-file-list">${state.docs.map(f => `<div class="knowledge-file-row"><button class="knowledge-file-open" data-action="preview-doc" data-id="${escapeHTML(f.id)}" ${k.uploading?'disabled':''} title="Preview ${escapeHTML(f.name)}"><span class="file-extension">${escapeHTML(f.name.split('.').pop().toUpperCase())}</span><span class="file-info"><strong>${escapeHTML(f.name)}</strong><small>${size(f.size)} · ${f.characters ? 'Ready for analysis' : 'Local reference'}</small></span>${icon('chevron')}</button><button class="icon-button knowledge-remove" data-action="remove-doc" data-id="${escapeHTML(f.id)}" ${k.uploading?'disabled':''} aria-label="Remove ${escapeHTML(f.name)}" title="Remove attachment">${icon('close')}</button></div>`).join('')}</div>` : `<button class="knowledge-empty dropzone" id="knowledge-dropzone" data-action="pick-knowledge" ${k.uploading?'disabled':''}>${icon('book')}<strong>No knowledge attached</strong><span>Drop files here or <u>browse files</u></span><small>PDF, Word (.docx), TXT or Markdown · up to 20 MB each</small></button>`;
  return `<dialog class="knowledge-dialog ${preview?'has-preview':''}" id="knowledge-dialog" aria-labelledby="knowledge-title">
    <header class="knowledge-header"><div><h2 id="knowledge-title">${preview ? 'Reference preview' : 'Knowledge'}</h2><p>${preview ? escapeHTML(preview.name) : 'Papers and notes for your next analysis.'}</p></div><button class="icon-button" data-action="close-knowledge" aria-label="Close knowledge">${icon('close')}</button></header>
    <div class="knowledge-body">${preview ? `<div class="knowledge-preview-bar"><button class="subtle-link" data-action="knowledge-list">${icon('left')}All files</button><span>Extracted text</span></div><pre class="knowledge-text">${escapeHTML(preview.text)}</pre><p class="knowledge-note">This is the text used by the model. Download the original to view the document layout.</p>` : list}
      ${k.uploading ? `<div class="knowledge-progress" role="status">${icon('spinner','spinner')}<span>${escapeHTML(k.progress || 'Preparing references…')}</span></div>` : ''}
      ${k.error ? `<p class="knowledge-error" role="alert">${escapeHTML(k.error)}</p>` : ''}
    </div><footer class="knowledge-footer">${preview ? button('Download original','download-doc','save',false,`data-id="${escapeHTML(preview.id)}"`) : button('Add files','pick-knowledge','plus',false,k.uploading?'disabled':'')}<small>${state.docs.length} ${state.docs.length===1?'file':'files'} attached</small>${button('Done','close-knowledge','',true)}</footer>
  </dialog>`;
}
function openKnowledge() { state.knowledge.open=true; state.knowledge.preview=null; render(); }
function closeKnowledge() { state.knowledge.open=false; state.knowledge.preview=null; render(); app.querySelector('[data-action="manage-knowledge"]')?.focus(); }

function submissionIssue() {
  const raw=String(state.featureNumber ?? '').trim(), count=Number(raw), c=state.config;
  if(!raw)return {title:'Feature number required',message:'Enter how many features you want MorphAgent to design before continuing.',target:'feature-number',label:'Enter feature number'};
  if(!/^\d+$/.test(raw) || !Number.isInteger(count) || count<1 || count>500)return {title:'Invalid feature number',message:'Feature number must be a whole number from 1 to 500.',target:'feature-number',label:'Edit feature number'};
  if(!state.dataset)return {title:'Dataset required',message:'Add your microscopy dataset or choose the Tau demo before continuing.',target:'data',label:'Back to Design'};
  if(!state.question.trim())return {title:'Question required',message:'Describe the biological question you want to investigate.',target:'question',label:'Enter your question'};
  return apiConnectionIssue(c.route);
}
function apiConnectionIssue(route) {
  const c=state.config;
  if(!c.apiKey.trim() && !c.hasApiKey)return {title:'API key required',message:'Add your API key in Model API settings. No analysis has started.',target:'api',label:'Open Model API'};
  if(!c.baseUrl.trim() || !c.model.trim())return {title:'Model connection incomplete',message:'Fill in the Base URL and Model in Model API settings.',target:'api',label:'Open Model API'};
  if(route!=='code' && !c.sameConnection) {
    if(!c.vlmApiKey.trim() && !c.hasVlmApiKey)return {title:'VLM API key required',message:'You selected a separate vision connection. Add its API key before running image scoring.',target:'api',label:'Open Model API'};
    if(!c.vlmBaseUrl.trim() || !c.vlmModel.trim())return {title:'VLM connection incomplete',message:'Fill in the separate VLM Base URL and vision model.',target:'api',label:'Open Model API'};
  }
  return null;
}
function showRunIssue(issue) {
  state.knowledge.open=false;
  state.runDialog={kind:'alert',...issue};render();
}
function showRunConfirmation(request) {
  state.runDialog={kind:'confirm',launching:false,request:JSON.parse(JSON.stringify(request)),summary:{
    question:request.question,featureNumber:request.featureNumber,mode:state.config.mode,route:state.config.route,
    dataset:{name:state.dataset.name||'Selected dataset',path:state.dataset.path||'',samples:state.dataset.summary?.sample_count,images:state.dataset.summary?.primary_image_count ?? state.dataset.count},
    knowledge:state.docs.map(d=>d.name)
  }};
  render();
}
function closeRunDialog(resolve=false) {
  const dialog=state.runDialog;if(!dialog || dialog.launching)return;
  state.runDialog=null;
  if(resolve && dialog.target==='api'){showSettings('api');return;}
  render();
  const field=resolve && ['question','feature-number'].includes(dialog.target) ? document.getElementById(dialog.target) : app.querySelector(state.page==='compute'?'[data-action="prepare-compute"]':'[data-action="prepare-run"]');
  field?.focus();
}
function runDialog() {
  const d=state.runDialog;if(!d)return '';
  if(d.kind==='delete-history')return `<dialog id="run-dialog" class="knowledge-dialog run-dialog history-dialog" aria-modal="true" aria-labelledby="run-dialog-title" aria-describedby="run-dialog-description"><header class="knowledge-header"><div><h2 id="run-dialog-title">Delete run?</h2><p id="run-dialog-description">Move this run and its exports to system Trash / Recycle Bin.</p></div><button class="icon-button" data-action="close-run-dialog" aria-label="Cancel deletion" ${d.launching?'disabled':''}>${icon('close')}</button></header><div class="run-review-body"><strong>${escapeHTML(d.name)}</strong>${(d.paths||[]).map(p=>`<p class="dataset-path">${escapeHTML(p)}</p>`).join('')}${d.preservedPaths?.length?`<p class="muted-note">External, shared or protected files will be kept:</p>${d.preservedPaths.map(p=>`<p class="dataset-path">${escapeHTML(p)}</p>`).join('')}`:''}<p class="muted-note">Original datasets are kept.</p></div><footer class="knowledge-footer run-dialog-footer">${button('Cancel','close-run-dialog','',false,d.launching?'disabled':'')}${button(d.launching?'Deleting…':'Delete run','confirm-delete-history','trash',true,d.launching?'disabled':'')}</footer></dialog>`;
  const alert=d.kind==='alert', s=d.summary, busy=d.launching, compute=d.workflow==='compute';
  const row=(label,value)=>`<div class="run-review-row"><dt>${label}</dt><dd>${value}</dd></div>`;
  let details='';
  if(!alert) {
    // Compute replays a fixed feature set, so there is no question to review;
    // the previous run's question is shown as the context VLM rescoring inherits.
    details=compute ? row('Mode','Compute · saved features')+row('Previous run',`<strong>${escapeHTML(s.source.name)}</strong>${s.source.question?`<small>${escapeHTML(s.source.question)}</small>`:''}<span class="review-path">${escapeHTML(s.source.path)}</span>`) : row('Biological question',`<p class="review-question">${escapeHTML(s.question)}</p>`)+`<div class="run-review-pair">${row('Mode',escapeHTML(modes[s.mode].name)+' · '+modes[s.mode].loops+' loops')}${row('Analysis route',escapeHTML(routes[s.route].name))}</div>`;
    details+=row('Data',`<strong>${escapeHTML(s.dataset.name)}</strong>${s.dataset.samples!=null?`<small>${s.dataset.samples} samples${s.dataset.images!=null?' · '+s.dataset.images+' images':''}</small>`:''}${s.dataset.path?`<span class="review-path">${escapeHTML(s.dataset.path)}</span>`:''}`);
    if(compute) details+=row('Selected features',`<strong class="review-count">${s.features.length}</strong><ul>${s.features.map(n=>'<li>'+escapeHTML(n)+'</li>').join('')}</ul>`);
    else details+=row('Knowledge',`${s.knowledge.length?`<ul>${s.knowledge.map(name=>`<li>${icon('book')}${escapeHTML(name)}</li>`).join('')}</ul>`:''}<label class="deep-research-option"><input id="deep-research-checkbox" type="checkbox" ${d.request.deepResearch?'checked':''}> Use Deep Research to prepare background knowledge</label>${s.knowledge.length?'<small>Deep Research is added alongside your uploaded files.</small>':''}`)+row('Feature number',`<strong class="review-count">${s.featureNumber}</strong><small>Target count · retained results may vary after validation.</small>`);
  }
  return `<dialog id="run-dialog" class="knowledge-dialog run-dialog ${alert?'run-alert':''}" role="${alert?'alertdialog':'dialog'}" aria-modal="true" aria-labelledby="run-dialog-title" aria-describedby="run-dialog-description">
    <header class="knowledge-header"><div>${alert?'<span class="run-alert-mark" aria-hidden="true">!</span>':''}<h2 id="run-dialog-title">${escapeHTML(alert?d.title:'Review run configuration')}</h2><p id="run-dialog-description">${escapeHTML(alert?d.message:'Check your inputs. Analysis starts only after you confirm.')}</p></div><button class="icon-button" data-action="close-run-dialog" aria-label="Close configuration" ${busy?'disabled':''}>${icon('close')}</button></header>
    ${alert?'':`<div class="run-review-body"><dl>${details}</dl></div>`}
    <footer class="knowledge-footer run-dialog-footer">${button(alert?'Cancel':'Back to edit','close-run-dialog','',false,busy?'disabled':'')}${alert?button(d.label||'OK','resolve-run-issue','',true):button(busy?'Starting analysis…':'Confirm and run','confirm-run',busy?'spinner':'run',true,busy?'disabled aria-busy="true"':'')}</footer>
  </dialog>`;
}

const previewFiles = new Map();
async function previewDoc(id) {
  const file=previewFiles.get(id), doc=state.docs.find(f=>f.id===id);
  if(!file || !doc) return;
  state.knowledge.preview={...doc,text:/\.(txt|md)$/i.test(doc.name) ? await file.text() : 'Document text is extracted in the running desktop app. Download the original to view this file.'};
  render();
}
function downloadDoc(id) {
  const file=previewFiles.get(id); if(!file)return;
  const url=URL.createObjectURL(file), a=document.createElement('a');
  a.href=url; a.download=file.name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function runPage() {
  const mode = modes[state.config.mode];
  const steps = [
    ['folder','Prepare images','Inspect the dataset, reusing masks or writing them when absent.'],
    ['spark','Plan features','Connect the biological question with visual hypotheses.'],
    ['both','Quantify features','Apply the selected code and VLM analysis routes.'],
    ['check','Validate and refine',`Review candidates across ${mode.loops} ${mode.loops === 1 ? 'loop' : 'loops'}.`],
  ];
  const isRunning = state.runStatus === 'running';
  const isDone = state.runStatus === 'complete';
  return `<section class="page fade-in">${pageHeading('Analysis run','A clear view of every step, from question to features.',isRunning ? button('Stop preview','stop-preview') : button(isDone ? 'View example features' : 'Preview run',isDone ? 'view-features' : 'start-preview',isDone ? 'right' : 'run',true))}<div class="run-layout"><div><div class="run-prompt">${escapeHTML(state.question || 'Generate biologically grounded features that describe Tau aggregation and neuronal structure.')}</div><div class="run-meta"><span>${icon('folder')}${escapeHTML(state.dataset?.name || 'Tau demo')}</span><span>${icon(routes[state.config.route].icon)}${routes[state.config.route].name}</span><span>${icon(mode.icon)}${mode.name} · ${mode.loops} ${mode.loops === 1 ? 'loop' : 'loops'}</span></div><div class="run-status"><span class="status-dot ${isRunning ? 'running' : ''}"></span>${isRunning ? 'Playing workflow preview…' : isDone ? 'Workflow preview complete' : 'Ready to preview'}</div><div class="timeline">${steps.map(([symbol,title,desc],i) => `<div class="timeline-item ${isDone || (isRunning && state.runStep > i) ? 'done' : isRunning && state.runStep === i ? 'current' : ''}"><span class="timeline-icon">${icon(isDone || (isRunning && state.runStep > i) ? 'check' : isRunning && state.runStep === i ? 'spinner' : symbol, isRunning && state.runStep === i ? 'spinner' : '')}</span><div class="timeline-copy"><strong>${title}</strong><p>${desc}</p></div></div>`).join('')}</div><p class="preview-note">UI simulation only. This preview does not run analysis or call a model API.</p></div><aside class="run-summary"><h2>RUN DETAILS</h2><div class="summary-row"><small>Discovery depth</small><strong>${mode.loops} ${mode.loops === 1 ? 'loop' : 'loops'}</strong></div><div class="summary-row"><small>External knowledge</small><strong>${state.docs.length ? `${state.docs.length} reference ${state.docs.length === 1 ? 'file' : 'files'}` : 'No files attached'}</strong></div><div class="summary-row"><small>Reproducibility</small><strong>Enabled</strong></div><img class="summary-thumb" src="assets/tau-wt.png" alt="Example Tau microscopy context" /><p class="preview-note">Tau demo image · sample visual context</p></aside></div></section>`;
}
function featureRows() {
  const query = state.search.toLowerCase();
  return features.filter(f => `${f.name} ${f.route} ${f.category}`.toLowerCase().includes(query)).map(f => `<label class="feature-row"><input type="checkbox" data-feature="${f.id}" ${state.selectedFeatures.has(f.id) ? 'checked' : ''} /><div><strong>${f.name}</strong><small>${f.description}</small></div><span class="route-tag">${f.route}</span><span>${f.category}</span></label>`).join('') || '<div class="empty">No features match your search.</div>';
}
function reusePage() {
  return `<section class="home compute-home fade-in"><div class="home-heading">${mark('discovery-mark')}<h1>Compute saved features</h1><p>Apply the features a previous run saved to a new dataset. The feature set is already fixed, so there is nothing to describe.</p></div><div class="composer composer-tools-only"><div class="composer-footer"><div class="composer-tools"><button class="composer-chip" data-action="compute-source">${icon('history')}Upload features</button><button class="composer-chip" data-action="reuse-upload">${icon('plus')}Add data</button></div><button class="submit" data-action="prepare-compute" aria-label="Review and compute">${icon('arrow')}</button></div></div><p class="compute-note">Launch the desktop workspace to load saved scripts and compute real measurements.</p></section>`;
}
function visualizePage() {
  const f = features[state.vizFeature];
  return `<section class="page fade-in">${pageHeading('See the evidence','Explore a feature alongside its visual context.')}<div class="viz-layout"><div><div class="section-eyebrow" style="padding-left:12px;margin-bottom:15px">EXAMPLE FEATURES</div>${features.slice(0,4).map((f,i) => `<button class="viz-feature ${state.vizFeature === i ? 'active' : ''}" data-action="viz-feature" data-index="${i}" aria-pressed="${state.vizFeature === i}"><strong>${f.name}</strong><small>${f.route} · ${f.category}</small></button>`).join('')}<div class="viz-info"><h2>${f.name}</h2><p>${f.description}</p></div></div><div><div class="image-viewer"><img src="assets/tau-${state.sample}.png" alt="${state.sample === 'wt' ? 'WT_1' : 'MU_1'} Tau microscopy image, shared sample context" /><div class="image-toolbar"><span>${icon('vision', 'small-icon')} Original image</span><button class="image-tab ${state.sample === 'wt' ? 'active' : ''}" data-action="sample" data-value="wt">WT_1</button><button class="image-tab ${state.sample === 'mu' ? 'active' : ''}" data-action="sample" data-value="mu">MU_1</button></div></div><p class="evidence-caption">Shared dataset context, not a feature-specific heatmap. Measurement results and validation evidence will appear here when the analysis pipeline is connected.</p></div></div></section>`;
}
function render() {
  const pages = { data: home, settings, compute: reusePage, visualize: visualizePage, help: helpPage };
  const active = document.activeElement;
  const focusAttrs = active?.dataset.action ? ['action','value','tab','key','index','id'].filter(k => active.dataset[k]).map(k => `[data-${k}="${active.dataset[k]}"]`).join('') : '';
  const body = render.lastPage === state.page ? pages[state.page]().replaceAll(' fade-in','') : pages[state.page]();
  app.innerHTML = `<div class="shell">${sidebar()}<main class="workspace">${header()}<div class="content">${body}</div></main></div>${knowledgeDialog()}${runDialog()}`;
  render.lastPage = state.page;
  const dialog=state.runDialog ? document.getElementById('run-dialog') : state.knowledge.open ? document.getElementById('knowledge-dialog') : null;
  if(dialog) { dialog.showModal(); dialog.addEventListener('cancel',e=>{e.preventDefault();state.runDialog?closeRunDialog():closeKnowledge();}); }
  if (focusAttrs) (dialog || app).querySelector(focusAttrs)?.focus({preventScroll:true});
  bindDropzone();
}
function navigate(page) { state.page = page==='run'?'data':page==='reuse'?'compute':page==='save'?'data':page; state.sidebarOpen = false; render(); window.scrollTo(0,0); }
function showSettings(tab = 'api') { if (state.page !== 'settings') state.previousPage = state.page; state.settingsTab = tab==='analysis'?'analysis':'api'; navigate('settings'); }
function toast(message) {
  const el = document.getElementById('toast');
  el.textContent = message; el.classList.add('visible');
  clearTimeout(toast.timer); toast.timer = setTimeout(() => el.classList.remove('visible'),3500);
}
function size(bytes) { return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1,Math.round(bytes / 1024))} KB`; }
function useDemo() {
  state.dataset = { name: 'Tau microscopy demo', demo: true };
  if (!state.question) state.question = 'Discover biologically grounded features that describe Tau aggregation and neuronal structure.';
}
function stopPreview() { clearTimeout(state.timer); state.timer = null; state.runStatus = 'idle'; state.runStep = 0; }
function startPreview() {
  stopPreview(); if (!state.dataset) useDemo(); state.runStatus = 'running'; state.runStep = 0; render();
  const tick = () => {
    state.runStep += 1;
    if (state.runStep >= 4) { state.runStatus = 'complete'; state.timer = null; }
    else state.timer = setTimeout(tick,1500);
    if (state.page === 'run') render();
  };
  state.timer = setTimeout(tick,1500);
}
function addDocs(files) {
  let added = 0, rejected = 0;
  for (const f of files) {
    if (!/\.(pdf|docx|txt|md)$/i.test(f.name)) { rejected++; continue; }
    if (!state.docs.some(d => d.name === f.name && d.size === f.size)) {
      const id=`preview-${Date.now()}-${state.docs.length}`;
      state.docs.push({id,name:f.name,size:f.size}); previewFiles.set(id,f); added++;
    }
  }
  state.knowledge.open=true; state.knowledge.preview=null;
  render();
  toast(rejected ? `${added} added. Supported formats: PDF, Word, TXT, and Markdown.` : `${added} reference ${added === 1 ? 'file' : 'files'} added to the preview.`);
}
function bindDropzone() {
  const zone = document.getElementById('knowledge-dropzone');
  if (!zone) return;
  zone.addEventListener('dragover',e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave',() => zone.classList.remove('drag-over'));
  zone.addEventListener('drop',e => { e.preventDefault(); if(!state.knowledge.uploading)addDocs(e.dataTransfer.files); });
}
function download(name,text,type) {
  const url = URL.createObjectURL(new Blob([text],{type}));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}
function exportConfig() {
  const c = state.config;
  download('morphagent-preview-config.json',JSON.stringify({
    design_preview: true, question: state.question, dataset: state.dataset,
    analysis: { method: c.route, mode: modes[c.mode].name, num_rounds: modes[c.mode].loops, reproduce: true },
    model: { base_url:c.baseUrl, name:c.model, same_connection_for_vlm:c.sameConnection, ...(!c.sameConnection ? {vlm_base_url:c.vlmBaseUrl,vlm_model:c.vlmModel} : {}) },
    knowledge: { enabled:state.docs.length>0, files:state.docs },
    selected_example_features:[...state.selectedFeatures],
  },null,2),'application/json'); toast('Configuration exported. API keys are excluded.');
}
function exportFeatures() {
  const cell = s => `"${String(s).replaceAll('"','""')}"`;
  const rows = features.filter(f => state.selectedFeatures.has(f.id)).map(f => [f.id,f.name,f.route,f.description,'Design preview only'].map(cell).join(','));
  download('morphagent-example-features.csv',['id,name,route,description,source',...rows].join('\r\n'),'text/csv;charset=utf-8');
  toast('Selected example feature definitions exported.');
}

app.addEventListener('input',e => {
  if (e.target.id === 'question') state.question = e.target.value;
  if (e.target.id === 'help-question') state.help.draft = e.target.value;
  if (e.target.id === 'feature-number') state.featureNumber = e.target.value;
  if (e.target.dataset.config) state.config[e.target.dataset.config] = e.target.value;
  if (e.target.id === 'feature-search') { state.search = e.target.value; document.getElementById('feature-list').innerHTML = featureRows(); }
});
app.addEventListener('change',e => {
  if(e.target.id==='deep-research-checkbox' && state.runDialog?.kind==='confirm')state.runDialog.request.deepResearch=e.target.checked;
  const feature = e.target.dataset.feature;
  if (feature) {
    e.target.checked ? state.selectedFeatures.add(feature) : state.selectedFeatures.delete(feature);
    document.getElementById('selection-count').textContent = `${state.selectedFeatures.size} ${state.selectedFeatures.size === 1 ? 'feature' : 'features'} selected`;
    document.getElementById('reuse-button').disabled = !state.selectedFeatures.size;
  }
});
app.addEventListener('keydown',e => {
  if (e.target.id === 'question' && e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); prepareRun(); }
  if (e.target.id === 'help-question' && e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendHelp(); }
});
function prepareRun() {
  if(state.knowledge.uploading || state.runDialog)return;
  const issue=submissionIssue();if(issue)return showRunIssue(issue);
  showRunConfirmation({question:state.question.trim(),featureNumber:Number(state.featureNumber)});
}
app.addEventListener('click',e => {
  const b = e.target.closest('[data-action]'); if (!b || b.disabled) return;
  const {action,page,tab,value,key,index,id} = b.dataset;
  if (action === 'navigate') navigate(page);
  else if (action === 'settings') showSettings(tab || 'api');
  else if (action === 'settings-tab') { state.settingsTab = tab; render(); }
  else if (action === 'back') navigate(state.previousPage);
  else if (action === 'sidebar') { state.sidebarOpen = !state.sidebarOpen; render(); }
  else if (action === 'new') { stopPreview(); state.previewSubmitted=false; state.question = ''; state.featureNumber=''; state.runDialog=null; state.dataset = null; state.historyOpen = false; navigate('data'); }
  else if (action === 'demo') { useDemo(); render(); toast('Tau demo added. Edit the question, then review your configuration.'); }
  else if (action === 'history') { useDemo(); state.historyOpen = true; navigate('visualize'); }
  else if (action === 'pick-data') document.getElementById('dataset-input').click();
  else if (action === 'manage-knowledge') openKnowledge();
  else if (action === 'close-knowledge') closeKnowledge();
  else if (action === 'knowledge-list') { state.knowledge.preview=null; render(); }
  else if (action === 'preview-doc') previewDoc(id);
  else if (action === 'download-doc') downloadDoc(id);
  else if (action === 'pick-knowledge') document.getElementById('knowledge-input').click();
  else if (action === 'mode') { state.config.mode = value; render(); }
  else if (action === 'route') { state.config.route = value; render(); }
  else if (action === 'toggle') { state.config[key] = !state.config[key]; render(); }
  else if (action === 'edit-api-key') { state.editingSecrets.add(key); render(); app.querySelector(`[data-config="${key}"]`)?.focus(); }
  else if (action === 'toggle-secret') { const input = b.previousElementSibling; input.type = input.type === 'password' ? 'text' : 'password'; b.setAttribute('aria-label',input.type === 'password' ? 'Show API key' : 'Hide API key'); }
  else if (action === 'remove-doc') { state.docs=state.docs.filter(d=>d.id!==id); previewFiles.delete(id); render(); }
  else if (action === 'save-settings') toast('Settings updated for this preview. No API connection was made.');
  else if (action === 'prepare-run') prepareRun();
  else if (action === 'help-send') sendHelp();
  else if (action === 'help-clear') { state.help={draft:'',messages:[],pending:false,error:''}; render(); }
  else if (['compute-source','reuse-upload','prepare-compute'].includes(action)) toast('Open the desktop workspace to compute with a saved run. This is an offline preview.');
  else if (action === 'close-run-dialog') closeRunDialog();
  else if (action === 'resolve-run-issue') closeRunDialog(true);
  else if (action === 'confirm-run' && state.runDialog?.kind==='confirm') { state.runDialog=null; state.previewSubmitted=true; navigate('data'); toast('UI preview only. No analysis or API call was started.'); }
  else if (action === 'start-preview') startPreview();
  else if (action === 'stop-preview') { stopPreview(); render(); }
  else if (action === 'view-features') navigate('reuse');
  else if (action === 'reuse-selected') { state.question = `Apply ${[...state.selectedFeatures].map(id => features.find(f => f.id === id).name.toLowerCase()).join(', ')} to the selected microscopy dataset.`; navigate('data'); toast(`${state.selectedFeatures.size} example features added to your question.`); }
  else if (action === 'viz-feature') { state.vizFeature = Number(index); render(); }
  else if (action === 'sample') { state.sample = value; render(); }
  else if (action === 'export-config') exportConfig();
  else if (action === 'export-features') exportFeatures();
});
document.getElementById('dataset-input').addEventListener('change',e => {
  const files = [...e.target.files]; if (!files.length) return;
  const images = files.filter(f => /\.(tiff?|png|jpe?g)$/i.test(f.name));
  if (!images.length) { toast('Choose a folder containing TIFF, PNG, or JPEG images.'); e.target.value = ''; return; }
  state.dataset = {name:files[0].webkitRelativePath.split('/')[0] || 'Selected dataset',count:images.length,demo:false};
  render(); toast(`${images.length} image files selected locally. Nothing was uploaded.`); e.target.value = '';
});
document.getElementById('knowledge-input').addEventListener('change',e => { addDocs(e.target.files); e.target.value = ''; });
document.addEventListener('keydown',e => {
  if(state.runDialog || state.knowledge.open)return;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); stopPreview(); state.question = ''; state.dataset = null; state.historyOpen = false; navigate('data'); document.getElementById('question')?.focus(); }
  if (e.key === 'Escape' && state.sidebarOpen) { state.sidebarOpen = false; render(); }
});
render();
