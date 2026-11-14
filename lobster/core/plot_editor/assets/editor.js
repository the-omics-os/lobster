'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const clone = v => JSON.parse(JSON.stringify(v));
  const canonical = v => Array.isArray(v) ? '[' + v.map(canonical).join(',') + ']' :
    v && typeof v === 'object' ? '{' + Object.keys(v).sort().map(k => JSON.stringify(k)+':'+canonical(v[k])).join(',')+'}' : JSON.stringify(v);
  const token = location.hash.slice(1) || sessionStorage.getItem('lobster-editor-token') || '';
  if (token) sessionStorage.setItem('lobster-editor-token', token);
  const initialRef = new URLSearchParams(location.search).get('ref');
  history.replaceState(null, '', location.pathname + location.search);
  const plot = $('plot');
  const surface = plot.parentElement;
  let figure, original, baseline, ref, config, busy = false, selected = null, requestId = null;
  let drawQueue = Promise.resolve(), drawing = false;
  let panelState = new Map();
  let timeline = null, polling = false;
  let previewSize = '';
  let activeView = 'figure', dataPage = null, dataRequest = 0;
  let pendingSave = null;

  function message(text, error = false) {
    const status = $('status');
    status.dataset.detail = text;
    status.textContent = text.startsWith('Loaded ') ? 'Revision loaded' : text.startsWith('Saved ') ? 'Revision saved' : text;
    status.title = text;
    status.className = error ? 'error' : '';
  }
  function updateDraftState() {
    const state = $('draft-state');
    state.textContent = busy ? 'Working…' : !figure ? 'Loading…' : dirty() ? 'Unsaved changes' : 'Saved';
    state.dataset.state = busy ? 'busy' : dirty() ? 'edited' : 'saved';
    $('reset').disabled = busy || activeView==='data' || !figure || !dirty();
  }
  function dirty() { return figure && (canonical(figure) !== baseline); }
  function freeze(value) { busy = value; for (const e of document.querySelectorAll('header,.document-bar,main,#data-view,footer')) e.inert = value; $('save').disabled = value || activeView==='data' || !dirty(); updateDraftState(); }
  function changed() { requestId = null; $('save').disabled = busy || activeView==='data' || !dirty(); updateDraftState(); message(dirty() ? 'Unsaved changes · Save creates a new revision' : 'No unsaved changes'); }
  async function api(path, body) {
    const response = await fetch('/api/' + path, { method: body ? 'POST' : 'GET',
      headers: { Authorization: 'Bearer ' + token, ...(body ? {'Content-Type':'application/json'} : {}) },
      ...(body ? {body:JSON.stringify(body)} : {}), cache:'no-store' });
    let data;
    try { data = await response.json(); } catch { throw Error('Editor server did not return a valid response. Reopen from Lobster.'); }
    if (!response.ok) {const error=Error(data.error || 'Editor request failed');error.status=response.status;throw error;}
    return data;
  }
  function render() {
    const display = clone(figure), selection = selected && {...selected};
    drawQueue = drawQueue.catch(() => {}).then(async () => {
      if(activeView==='data') return;
      drawing = true;
      try {
        // Preserve fixed-size figure composition and scale the entire preview.
        // Reflowing a publication layout can crowd legends and paper annotations.
        const viewportWidth=Math.max(1,surface.clientWidth), viewportHeight=Math.max(1,surface.clientHeight);
        const template=display.layout.template?.layout || {};
        const dimension=(value,fallback)=>typeof value==='number' && Number.isFinite(value) && value>=10 ? value : fallback;
        const width=dimension(display.layout.width ?? template.width,Math.max(640,viewportWidth));
        const height=dimension(display.layout.height ?? template.height,Math.max(480,viewportHeight));
        const scale=Math.min(viewportWidth/width,viewportHeight/height,1);
        Object.assign(plot.style,{width:width+'px',height:height+'px',left:(viewportWidth-width*scale)/2+'px',top:(viewportHeight-height*scale)/2+'px',transform:`scale(${scale})`});
        previewSize=viewportWidth+':'+viewportHeight;
        function fit(layout) {
          layout.width=width; layout.height=height; layout.autosize=false;
        }
        fit(display.layout);
        for(const frame of display.frames || []) fit(frame.layout??={});
        if (selection) display.data[selection.trace].selectedpoints = [selection.point];
        await Plotly.react(plot, display.data, display.layout, {...config, responsive:false, autosizable:false, displaylogo:false, showLink:false, showSendToCloud:false, modeBarButtonsToRemove:[...(config.modeBarButtonsToRemove || []),'toImage']});
        if (plot._transitionData?._frames?.length) await Plotly.deleteFrames(plot);
        if (display.frames?.length) await Plotly.addFrames(plot, display.frames);
      } catch (e) { message('Cannot render working figure: '+e.message, true); }
      finally { drawing = false; }
    });
    return drawQueue;
  }
  function edit(fn, rebuild = false) {
    if (busy) return;
    selected = null; fn(); changed();
    if(rebuild) $('controls').inert=true;
    const result=render();
    if(rebuild) result.then(()=>{controls();$('controls').inert=false;});
    return result;
  }
  function object(parent, key) { if (!parent[key] || typeof parent[key] !== 'object' || Array.isArray(parent[key])) parent[key] = {}; return parent[key]; }
  function titleText(v) { return typeof v === 'string' ? v : v?.text || ''; }
  function color(v, fallback='#ffffff') {
    if(typeof v !== 'string' || !CSS.supports('color',v) || v === 'transparent') return fallback;
    const context=document.createElement('canvas').getContext('2d');
    context.fillStyle=v;
    return /^#[a-f\d]{6}$/i.test(context.fillStyle) ? context.fillStyle : fallback;
  }
  function section(title, open = true, parent = $('controls')) {
    const e = document.createElement('details');
    const key=(parent.dataset.section ? parent.dataset.section+'/' : '')+title;
    e.dataset.section=key; e.open = panelState.has(key) ? panelState.get(key) : open;
    const summary = document.createElement('summary'); summary.textContent = title; e.append(summary); parent.append(e); return e;
  }
  function field(parent, text, type, value, apply, options = []) {
    const label = document.createElement('label'), caption = document.createElement('span'); caption.textContent = text;
    label.className='field'; label.dataset.kind=type;
    const input = document.createElement(type === 'select' ? 'select' : 'input'); input.setAttribute('aria-label', text);
    if (type === 'select') {
      for (const option of options) { const [v,t] = Array.isArray(option) ? option : [option,option]; const e=document.createElement('option'); e.value=v; e.textContent=t; input.append(e); }
      if (value != null && !options.some(o => String(Array.isArray(o)?o[0]:o) === String(value))) {
        const e=document.createElement('option'); e.value=value; e.textContent=text==='Font'?String(value).split(',')[0].replace(/[\"']/g,'').trim():value; e.title=value; input.append(e);
      }
      input.value=value;
    } else { input.type=type; if (type === 'checkbox') input.checked=!!value; else input.value=value??''; }
    if (type === 'number') input.step='any';
    input.addEventListener(['text','color'].includes(type) ? 'input' : 'change', () => {
      const v=type==='checkbox' ? input.checked : type==='number' ? Number(input.value) : input.value;
      if (type==='number' && (!input.value || !Number.isFinite(v))) return;
      edit(() => apply(v));
    });
    label.append(caption);
    if(type==='color') {
      const control=document.createElement('span'), valueText=document.createElement('span');
      control.className='color-control'; valueText.className='color-value'; valueText.textContent=input.value;
      input.addEventListener('input',()=>valueText.textContent=input.value);
      control.append(input,valueText); label.append(control);
    } else label.append(input);
    parent.append(label); return input;
  }
  function button(parent, text, action) { const e=document.createElement('button'); e.textContent=text; e.onclick=action; parent.append(e); return e; }
  function note(parent, text) { const e=document.createElement('p'); e.textContent=text; parent.append(e); }
  const cartesianTypes = ['scatter','scattergl','bar','histogram','box','violin','heatmap','contour'];
  function axes() {
    const keys=new Set(Object.keys(figure.layout).filter(k => /^[xy]axis\d*$/.test(k)));
    for (const trace of figure.data) if (cartesianTypes.includes(trace.type || 'scatter')) {
      keys.add((trace.xaxis || 'x').replace('x','xaxis')); keys.add((trace.yaxis || 'y').replace('y','yaxis'));
    }
    return [...keys];
  }
  function controls() {
    for(const detail of $('controls').querySelectorAll('details')) panelState.set(detail.dataset.section,detail.open);
    $('controls').replaceChildren(); const layout=figure.layout;
    const effectiveFont={...plot._fullLayout?.font,...layout.template?.layout?.font,...layout.font};
    const font=section('Typography');
    field(font,'Font','select',effectiveFont.family || 'Arial',v=>object(layout,'font').family=v,['Arial','Helvetica','Georgia','Times New Roman','Courier New']);
    field(font,'Font size','number',effectiveFont.size || 14,v=>object(layout,'font').size=Math.max(1,v));
    const colors=section('Colors',false);
    field(colors,'Text color','color',color(effectiveFont.color,'#263e33'),v=>object(layout,'font').color=v);
    field(colors,'Canvas color','color',color(layout.paper_bgcolor ?? plot._fullLayout?.paper_bgcolor),v=>layout.paper_bgcolor=v);
    field(colors,'Plot background','color',color(layout.plot_bgcolor ?? plot._fullLayout?.plot_bgcolor),v=>layout.plot_bgcolor=v);
    for (const key of axes()) {
      const a=layout[key] || {}, parent=section(key.replace('axis',' axis').replace(/^[xy]/,v=>v.toUpperCase()),false);
      field(parent,'Axis title','text',titleText(a.title),v=>object(object(layout,key),'title').text=v);
      field(parent,'Scale','select',a.type || 'linear',v=>object(layout,key).type=v,['linear','log','date','category']);
      for (const [prop,label,fallback] of [['showgrid','Grid',true],['zeroline','Zero line',true],['showticklabels','Tick labels',true],['autorange','Auto range',true]])
        field(parent,label,'checkbox',a[prop]??fallback,v=>object(layout,key)[prop]=v);
      for (const [index,label] of [[0,'Range start'],[1,'Range end']]) field(parent,label,'text',a.range?.[index]??'',v=> {
        const axis=object(layout,key); axis.range=clone(axis.range || [null,null]); axis.range[index]=v!=='' && !isNaN(Number(v)) ? Number(v) : v; axis.autorange=false;
      });
    }
    const traces=section('Traces'); traces.className='trace-list';
    const count=document.createElement('span');count.className='section-count';count.textContent=figure.data.length;
    traces.querySelector('summary').append(count);
    figure.data.forEach((t,i) => {
      const parent=section(t.name || `Trace ${i+1}`,false,traces), type=t.type || 'scatter';
      const summary=parent.querySelector('summary'); summary.replaceChildren();
      const dot=document.createElement('span'), name=document.createElement('span'), caption=document.createElement('span');
      dot.className='trace-dot';dot.style.backgroundColor=color(typeof t.marker?.color==='string'?t.marker.color:plot._fullData?.[i]?.marker?.color,'#8994a3');
      name.textContent=t.name || `Trace ${i+1}`;caption.className='trace-caption';
      const points=array(t.x);caption.textContent=points?`${points.length} points`:type;
      summary.append(dot,name,caption);
      field(parent,'Trace name','text',t.name || '',v=>{t.name=v;name.textContent=v || `Trace ${i+1}`;});
      field(parent,'Visible','checkbox',t.visible!==false && t.visible!=='legendonly',v=>t.visible=v);
      field(parent,'Opacity','number',t.opacity??1,v=>t.opacity=Math.max(0,Math.min(1,v)));
      if (['scatter','scattergl'].includes(type)) {
        field(parent,'Drawing','select',t.mode || 'lines+markers',v=>t.mode=v,['lines','markers','lines+markers','markers+text','lines+markers+text']);
        field(parent,'Marker size','number',typeof t.marker?.size==='number'?t.marker.size:7,v=>object(t,'marker').size=Math.max(0,v));
        field(parent,'Marker color','color',color(t.marker?.color,'#18816b'),v=>object(t,'marker').color=v);
        field(parent,'Marker symbol','select',typeof t.marker?.symbol==='string'?t.marker.symbol:'circle',v=>object(t,'marker').symbol=v,['circle','square','diamond','cross','x']);
        field(parent,'Line width','number',t.line?.width??2,v=>object(t,'line').width=Math.max(0,v));
        field(parent,'Line color','color',color(t.line?.color,'#18816b'),v=>object(t,'line').color=v);
      }
      if (type==='bar') field(parent,'Bar color','color',color(t.marker?.color,'#18816b'),v=>object(t,'marker').color=v);
      if (['heatmap','contour','surface'].includes(type)) field(parent,'Color scale','select',typeof t.colorscale==='string'?t.colorscale:'Viridis',v=>t.colorscale=v,['Viridis','Cividis','Plasma','Blues','RdBu']);
    });
    const layoutControls=section('Legend & layout',false);
    field(layoutControls,'Show legend','checkbox',layout.showlegend!==false,v=>layout.showlegend=v);
    field(layoutControls,'Legend direction','select',layout.legend?.orientation || 'v',v=>object(layout,'legend').orientation=v,[['v','Vertical'],['h','Horizontal']]);
    for (const [side,label] of [['l','Left margin'],['r','Right margin'],['t','Top margin'],['b','Bottom margin']])
      field(layoutControls,label,'number',layout.margin?.[side]??50,v=>object(layout,'margin')[side]=Math.max(0,v));
    if (figure.data.some(t=>t.type==='bar')) field(layoutControls,'Bar layout','select',layout.barmode || 'group',v=>layout.barmode=v,['group','stack','relative','overlay']);
    if (axes().length) annotations();
    entities();
  }
  function annotations() {
    const parent=section('Annotations',false), layout=figure.layout;
    const label=document.createElement('label'); label.className='field';label.dataset.kind='text';const caption=document.createElement('span');caption.textContent='New label'; const input=document.createElement('input'); input.setAttribute('aria-label','New annotation'); label.append(caption,input); parent.append(label);
    button(parent,'Add at center',()=> { if(input.value) edit(()=>{ (layout.annotations??=[]).push({text:input.value,x:.5,y:.5,xref:'paper',yref:'paper',showarrow:false}); },true); });
    (layout.annotations || []).forEach((a,i) => {
      field(parent,`Annotation ${i+1}`,'text',a.text || '',v=>a.text=v);
      for (const key of ['x','y']) field(parent,`Label ${i+1} ${key}`,'text',a[key]??'',v=>a[key]=v!=='' && !isNaN(Number(v))?Number(v):v);
      button(parent,`Remove label ${i+1}`,()=>edit(()=>layout.annotations.splice(i,1),true));
    });
  }
  function array(v) {
    if(Array.isArray(v)) return v;
    if(!v || typeof v.bdata!=='string' || (v.shape && String(v.shape).includes(','))) return null;
    const Type={f4:Float32Array,f8:Float64Array,i1:Int8Array,i2:Int16Array,i4:Int32Array,u1:Uint8Array,u2:Uint16Array,u4:Uint32Array}[v.dtype];
    if(!Type) return null;
    try { return Array.from(new Type(Uint8Array.from(atob(v.bdata),c=>c.charCodeAt(0)).buffer)); } catch { return null; }
  }
  function pointIds(t) {
    if(Array.isArray(t.ids)) return t.ids;
    const i=t.meta?.lobster_ids?.customdata_index;
    return Number.isInteger(i) && i>=0 && Array.isArray(t.customdata) ? t.customdata.map(r=>Array.isArray(r)?r[i]:null) : null;
  }
  function entities() {
    const index=[];
    figure.data.forEach((t,trace)=> {
      if(!['scatter','scattergl'].includes(t.type || 'scatter')) return;
      const ids=pointIds(t),x=array(t.x),y=array(t.y);
      if(!ids || !x || !y || ids.length!==x.length || ids.length!==y.length || new Set(ids).size!==ids.length || ids.some(id=>typeof id!=='string'||!id)) return;
      ids.forEach((id,point)=>index.push({id,trace,point,x:x[point],y:y[point],count:ids.length}));
    });
    if(!index.length) return;
    const parent=section('Find entities'), input=document.createElement('input'); input.placeholder='Search point IDs…'; input.setAttribute('aria-label','Search point IDs'); parent.append(input);
    const list=document.createElement('div'); list.className='results'; parent.append(list);
    const actions=document.createElement('div'); actions.className='button-row'; parent.append(actions);
    const update=()=> {
      list.replaceChildren();
      for(const point of index.filter(p=>p.id.toLowerCase().includes(input.value.toLowerCase())).slice(0,60)) {
        const b=button(list,point.id+' · '+(figure.data[point.trace].name || `Trace ${point.trace+1}`),()=>{ selected=point; update(); render(); });
        b.classList.toggle('active',selected?.trace===point.trace && selected?.point===point.point);
      }
      for (const b of actions.querySelectorAll('button')) b.disabled=!selected;
    };
    button(actions,'Clear highlight',()=>{selected=null;update();render();});
    button(actions,'Label selected',()=> {
      const p=selected; if(!p || p.x==null || p.y==null || typeof p.x==='object' || typeof p.y==='object') return;
      const trace=figure.data[p.trace]; edit(()=>{(figure.layout.annotations??=[]).push({text:p.id,x:p.x,y:p.y,xref:trace.xaxis || 'x',yref:trace.yaxis || 'y',showarrow:true});});update();
    });
    for(const [name,hide] of [['Hide marker',true],['Restore marker',false]]) button(actions,name,()=> {
      const p=selected; if(!p) return;
      const t=figure.data[p.trace]; if(t.mode && !t.mode.includes('markers')) {message('This trace has no markers to hide.',true);return;}
      edit(()=> {
        const marker=object(t,'marker');
        if(!Array.isArray(marker.opacity)) marker.opacity=array(marker.opacity) || Array.from({length:p.count},()=>typeof marker.opacity==='number'?marker.opacity:1);
        const old=original.data[p.trace]?.marker?.opacity, oldValues=array(old);
        marker.opacity[p.point]=hide?0:(oldValues?oldValues[p.point]:typeof old==='number'?old:1);
      }); update();
    });
    note(parent,'Up to 60 matching IDs. Hidden markers remain in the source data.'); input.oninput=update; update();
  }
  async function historyList(latest = null) {
    timeline=latest || await api('history'); const rows=timeline.revisions; $('history').replaceChildren();
    $('delete-revision').disabled=!ref || ref.split('@')[1]===timeline.head;
    $('delete-revision').title=ref?.split('@')[1]===timeline.head?'The original revision is pinned.':'Delete the loaded revision; neighboring versions are kept.';
    $('revision-count').textContent=rows.length+' '+(rows.length===1?'revision':'revisions');
    rows.forEach((row,i)=> {
      const target=row.figure_id+'@'+row.revision_id;
      const e=button($('history'),'',()=>load(target)); e.title=target+' · parents: '+(row.parents?.join(', ') || 'original');
      const head=document.createElement('span'),name=document.createElement('span');head.className='revision-head';name.className='revision-name';name.textContent=(row.revision_id===timeline.head?'Original':`Revision ${i+1}`)+(row.revision_id===timeline.tail?' · Latest':'');head.append(name);
      if(target===ref) e.setAttribute('aria-current','true');
      e.append(head);
      e.classList.toggle('active',target===ref); e.disabled=row.availability!=='available';
      const small=document.createElement('small'),date=new Date(row.created_at);
      small.textContent=Number.isNaN(date.getTime())?'Date unavailable':new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}).format(date);
      e.append(small);
    });
    revisionChanges(rows.find(row=>row.ref===ref));
  }
  function revisionChanges(row) {
    const panel=$('revision-changes'), text=$('revision-change-text');
    const change=row?.change;
    const description=change?.request || change?.summary || '';
    panel.hidden=!description; text.textContent=description;
    const list=$('revision-change-properties'); list.replaceChildren();
    for(const item of change?.details?.properties || []) {
      const entry=document.createElement('li');
      entry.textContent=item.path+': '+JSON.stringify(item.before)+' → '+JSON.stringify(item.after);
      list.append(entry);
    }
    $('revision-change-detail').hidden=!list.childElementCount;
  }
  async function loadData(offset=0) {
    const target=ref, request=++dataRequest;
    dataPage=null;$('data-table').tHead.replaceChildren();$('data-table').tBodies[0].replaceChildren();
    $('data-count').textContent='';$('data-range').textContent='';$('data-operations').hidden=true;
    $('data-notice').textContent='Loading saved drawing source…'; $('data-notice').className='data-notice';
    $('data-previous').disabled=true; $('data-next').disabled=true;
    try {
      const page=await api('data?ref='+encodeURIComponent(target)+'&offset='+offset+'&limit=100');
      if(request!==dataRequest || target!==ref || activeView!=='data') return;
      dataPage=page;
      const head=$('data-table').tHead, body=$('data-table').tBodies[0]; head.replaceChildren();body.replaceChildren();
      $('data-count').textContent=''; $('data-range').textContent=''; $('data-operations').hidden=true;
      if(page.status!=='available') { $('data-notice').textContent=page.message; return; }
      $('data-count').textContent=page.total_rows.toLocaleString()+' rows · '+page.columns.length+' columns';
      $('data-notice').textContent=(page.mapping_status==='available'?'Highlighted headers show recorded column roles.':'Column mapping unavailable for this revision; no roles have been inferred.')+
        ' This is the retained drawing table, not a live view of the uploaded workbook.'+(dirty()?' Unsaved figure edits are not reflected here.':'');
      const descriptions=(page.transformations || []).map(item=>item.description || item.operation).filter(v=>typeof v==='string');
      if(page.selection?.method) descriptions.push('Row selection: '+page.selection.method+(page.selection.column?' · '+page.selection.column+' = '+String(page.selection.equals):''));
      if(page.selection?.sampled) descriptions.push(page.selection.plotted_rows+' of '+page.selection.input_rows+' input rows retained for plotting.');
      $('data-operations').textContent=descriptions.join(' ');$('data-operations').hidden=!descriptions.length;
      const header=document.createElement('tr'), corner=document.createElement('th');corner.textContent='#';corner.className='row-number';corner.scope='col';header.append(corner);
      const names={x:'X',y:'Y',z:'Z',color:'Color / group',size:'Size',id:'Identifier',label:'Label',filter:'Filter'};
      for(const column of page.columns) {
        const cell=document.createElement('th');cell.scope='col';cell.classList.toggle('column-used',column.roles.length>0);
        const name=document.createElement('span');name.className='column-name';name.textContent=column.name;cell.append(name);
        for(const role of column.roles.length?column.roles:[{role:page.mapping_status==='available'?'unused':'unknown'}]) {
          const badge=document.createElement('span');badge.className='column-role';badge.dataset.role=role.role;
          badge.textContent=names[role.role] || (role.role==='unused'?'Unused':'Role unknown');
          if(role.transform) {badge.textContent+=' · transformed';badge.title=String(role.transform);}
          cell.append(badge);
        }
        header.append(cell);
      }
      head.append(header);
      page.rows.forEach((values,index)=>{
        const row=document.createElement('tr'), number=document.createElement('th');number.scope='row';number.className='row-number';number.textContent=page.offset+index+1;row.append(number);
        for(const value of values) {
          const cell=document.createElement('td');cell.textContent=value===null?'null':typeof value==='object'?JSON.stringify(value):String(value);cell.title=cell.textContent;
          cell.classList.toggle('numeric',typeof value==='number');cell.classList.toggle('null',value===null);row.append(cell);
        }
        body.append(row);
      });
      $('data-range').textContent=page.total_rows?`${page.offset+1}–${page.offset+page.rows.length} of ${page.total_rows.toLocaleString()} rows`:'No rows';
      $('data-previous').disabled=page.offset===0;$('data-next').disabled=page.offset+page.rows.length>=page.total_rows;
      $('data-table').parentElement.scrollTop=0;
    } catch(e) {
      if(request===dataRequest && activeView==='data') {$('data-notice').textContent=e.message;$('data-notice').className='data-notice error';}
    }
  }
  async function switchView(view) {
    if(busy || view===activeView) return;
    activeView=view;dataRequest++;
    for(const name of ['figure','data']) {
      $(name+'-view').hidden=name!==view;
      $(name+'-tab').setAttribute('aria-selected',String(name===view));$(name+'-tab').tabIndex=name===view?0:-1;
    }
    document.querySelector('.document-actions').hidden=view==='data';$('title').readOnly=view==='data';
    $('save').disabled=view==='data' || !dirty();updateDraftState();
    if(view==='data') { if(ref) await loadData(0); }
    else if(figure) { await render();controls(); }
  }
  for(const name of ['figure','data']) {
    $(name+'-tab').onclick=()=>switchView(name);
    $(name+'-tab').onkeydown=event=>{
      if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
        event.preventDefault();const next=event.key==='Home'?'figure':event.key==='End'?'data':name==='figure'?'data':'figure';
        $(next+'-tab').focus();switchView(next);
      }
    };
  }
  $('data-previous').onclick=()=>{if(dataPage) loadData(Math.max(0,dataPage.offset-100));};
  $('data-next').onclick=()=>{if(dataPage) loadData(dataPage.offset+100);};
  async function load(target, ask = true) {
    if(busy) return;
    if(ask && dirty() && !confirm('Discard unsaved changes and load this revision?')) return;
    freeze(true);
    try {
      const data=await api('figure'+(target?'?ref='+encodeURIComponent(target):''));
      figure=data.figure; figure.layout??={}; original=clone(figure); baseline=canonical(figure); ref=data.ref; config=data.config; selected=null;requestId=null;
      pendingSave=null;
      $('title').value=titleText(figure.layout.title); $('preset').value='Original'; await render(); controls(); await historyList();
      dataPage=null;if(activeView==='data') await loadData(0);
      const scatter=figure.data.every(t=>['scatter','scattergl'].includes(t.type || 'scatter'));
      const points=scatter?figure.data.reduce((n,t)=>n+(array(t.x)?.length || 0),0):null;
      $('figure-meta').textContent=(points!==null?points.toLocaleString()+' points · ':'')+figure.data.length+' '+(figure.data.length===1?'trace':'traces');
      const next=new URL(location.href); next.searchParams.set('ref',ref); history.replaceState(null,'',next.pathname+next.search); message('Loaded '+ref);
    } catch(e) { message(e.message,true); }
    finally {freeze(false);}
  }
  $('title').oninput=()=>edit(()=>object(figure.layout,'title').text=$('title').value);
  $('preset').onchange=()=> {
    const style=$('preset').value;
    edit(()=> {
      if(style==='Original') figure=clone(original);
      else {
        const layout=figure.layout;
        layout.template={}; layout.paper_bgcolor='#ffffff'; layout.plot_bgcolor='#ffffff';
        layout.font={family:style==='Publication'?'Georgia':'Arial',size:style==='Presentation'?22:style==='Publication'?14:13,color:'#243d32'};
        for(const key of axes()) Object.assign(object(layout,key),{showgrid:style==='Presentation',gridcolor:'#e6ede8',zeroline:false,showline:style==='Publication',ticks:style==='Publication'?'outside':''});
        layout.margin={l:65,r:40,t:70,b:65};
      }
      $('title').value=titleText(figure.layout.title);
    },true);
  };
  $('reset').onclick=()=>{if(figure && (!dirty() || confirm('Discard unsaved changes?'))) edit(()=>{pendingSave=null;figure=clone(original);$('title').value=titleText(figure.layout.title);$('preset').value='Original';},true);};
  $('save').onclick=async()=> {
    if(!dirty() || busy) return;
    freeze(true); message('Validating and saving revision…');
    requestId??=crypto.randomUUID().replaceAll('-','');
    let submitted=false;
    try {
      const draft=pendingSave??={parent:ref,figure:clone(figure),request_id:requestId,generation:timeline.generation,
                   explanation:'Manual visual edit'};
      await drawQueue; await api('validate',draft); submitted=true; const result=await api('save',draft);
      if(result.status!=='available') {
        if(result.status!=='pending') pendingSave=null;
        throw Error((result.status==='pending'?'Save confirmation pending; retry this same draft':'Save failed')+': '+(result.reason || result.status));
      }
      // A retry may confirm an older draft after the user made additional edits.
      // Keep those newer edits unsaved instead of treating them as committed.
      ref=result.ref; original=clone(draft.figure); baseline=canonical(original);requestId=null;selected=null;pendingSave=null;
      const next=new URL(location.href);next.searchParams.set('ref',ref);history.replaceState(null,'',next.pathname+next.search);
      await historyList(); await render();message(dirty()?'Previous save confirmed. Newer edits are still unsaved.':'Saved '+ref);
    } catch(e) {
      if(!submitted || e.status===409) {pendingSave=null;requestId=null;}
      message(e.message+(pendingSave?' · Retry Save revision to confirm this save; newer edits stay unsaved.':''),true);
    } finally {freeze(false);}
  };
  $('refresh-history').onclick=async()=>{try {await historyList();requestId=null;message('History refreshed. Your draft is preserved.');}catch(e){message(e.message,true);}};
  $('delete-revision').onclick=async()=> {
    if(busy || !timeline || ref.split('@')[1]===timeline.head) return;
    if(!confirm('Delete this revision from the timeline?'+(dirty()?' Unsaved changes will be discarded.':''))) return;
    freeze(true);
    let target=null;
    try {const result=await api('delete',{ref,generation:timeline.generation});target=result.load_ref;}
    catch(e){message(e.message,true);} finally {freeze(false);}
    if(target) await load(target,false);
  };
  function assignPath(parent,key,value) {
    const parts=key.replace(/\[(\d+)\]/g,'.$1').split('.');
    if(parts.some(p=>['__proto__','prototype','constructor'].includes(p))) return;
    for(const part of parts.slice(0,-1)) parent=parent[part]??={}; parent[parts.at(-1)]=clone(value);
  }
  const narrow=matchMedia('(max-width: 800px)');
  function showProperties(show) {
    $('inspector').hidden=!show;
    document.body.classList.toggle('inspector-hidden',!show);
    $('properties-toggle').setAttribute('aria-expanded',String(show));
  }
  $('properties-toggle').onclick=()=>showProperties($('inspector').hidden);
  narrow.addEventListener('change',()=>showProperties(!narrow.matches));
  showProperties(!narrow.matches);
  window.addEventListener('beforeunload',e=>{if(dirty()){e.preventDefault();e.returnValue='';}});
  let resizeFrame;
  new ResizeObserver(()=>{
    cancelAnimationFrame(resizeFrame);
    resizeFrame=requestAnimationFrame(()=>{
      if(figure && previewSize!==surface.clientWidth+':'+surface.clientHeight) render();
    });
  }).observe(surface);
  async function syncHistory() {
    if(busy || polling || !timeline || document.hidden) return;
    polling=true;
    try {
      const previous=timeline, latest=await api('history');
      if(busy || latest.generation===timeline.generation) return;
      const followedLatest=ref===previous.figure_id+'@'+previous.tail;
      await historyList(latest);
      if(dirty()) message('History updated. Your unsaved changes are preserved.');
      else if(followedLatest && latest.tail && ref!==latest.figure_id+'@'+latest.tail)
        await load(latest.figure_id+'@'+latest.tail,false);
      else if(!latest.nodes[ref.split('@')[1]])
        await load(latest.figure_id+'@'+latest.tail,false);
    } catch(e) { message(e.message,true); }
    finally { polling=false; }
  }
  setInterval(syncHistory,3000);
  setInterval(()=>api('heartbeat').catch(()=>message('Editor session ended. Reopen it from Lobster.',true)),30000);
  load(initialRef,false).then(()=> {
    if(!plot.on) return;
    plot.on('plotly_relayout',changes=> {
      if(drawing || busy || !figure) return;
      for(const [key,value] of Object.entries(changes)) if(!['width','height','autosize'].includes(key)) assignPath(figure.layout,key,value);
      changed();
    });
    plot.on('plotly_restyle',([changes,indices])=> {
      if(drawing || busy || !figure) return;
      for(const [key,value] of Object.entries(changes)) indices.forEach((trace,i)=>assignPath(figure.data[trace],key,Array.isArray(value)?value[i%value.length]:value));
      changed();
    });
  });
})();
