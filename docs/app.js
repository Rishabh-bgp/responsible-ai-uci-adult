const $=s=>document.querySelector(s),f3=v=>v==null||isNaN(v)?'—':(+v).toFixed(3);
const SHORT={'Logistic Regression':'LogReg','Random Forest':'Rand. Forest','XGBoost':'XGBoost','Neural Network (MLP)':'MLP'};
function seg(el,opts,cb,init){el.innerHTML='';opts.forEach(o=>{const b=document.createElement('button');b.textContent=o;b.type='button';b.onclick=()=>{[...el.children].forEach(c=>c.classList.remove('on'));b.classList.add('on');cb(o)};el.appendChild(b);if(o===init)b.classList.add('on')});cb(init)}
function table(el,cols,rows,hl){let h='<table><tr>'+cols.map(c=>`<th>${c[1]}</th>`).join('')+'</tr>';
 const best={};(hl||[]).forEach(k=>best[k]=Math.max(...rows.map(r=>r[k])));
 rows.forEach(r=>{h+='<tr>'+cols.map(([k])=>{const v=r[k];const cls=hl&&hl.includes(k)&&v===best[k]?' class="best"':'';return `<td${cls}>${typeof v==='number'?f3(v):(v??'—')}</td>`}).join('')+'</tr>'});el.innerHTML=h+'</table>'}
fetch('data/results.json').then(r=>r.json()).then(D=>{
 $('#nrows').textContent=D.n.toLocaleString();$('#nboot').textContent=D.n_boot.toLocaleString();
 // 01
 const M=['accuracy','precision','recall','f1','roc_auc'];
 seg($('#metricSeg'),M,m=>{const lo=Math.min(...D.table3.map(r=>r[m]))-.05;$('#perfBars').innerHTML=D.table3.map(r=>`<div class="row"><span>${r.model}</span><div class="bar"><div style="width:${(r[m]-lo)/(1-lo)*100}%"></div></div><b>${f3(r[m])}</b></div>`).join('')+`<p class="note">Bar axis starts at ${f3(lo)} to make differences visible.</p>`},'roc_auc');
 table($('#t3'),[['model','Model'],['threshold','Thr.'],...M.map(m=>[m,m.toUpperCase().replace('_','-')])],D.table3,M);
 // 02
 seg($('#gapSeg'),['DPD','EOD','FPRD'],k=>{const W=720,H=260,x0=150,xs=v=>x0+v/0.35*(W-x0-30);let s='';
  [0,.1,.2,.3].forEach(t=>s+=`<line x1="${xs(t)}" x2="${xs(t)}" y1="10" y2="${H-25}" stroke="#262638"/><text x="${xs(t)-8}" y="${H-8}">${t.toFixed(1)}</text>`);
  D.table5.forEach((r,i)=>{const y=40+i*55,v=r[k];s+=`<text x="4" y="${y+4}">${r.model}</text><line x1="${xs(v.lo)}" x2="${xs(v.hi)}" y1="${y}" y2="${y}" stroke="#2ee6d6" stroke-width="3"/><rect x="${xs(v.lo)}" y="${y-9}" width="${xs(v.hi)-xs(v.lo)}" height="18" fill="rgba(46,230,214,.12)"/><circle cx="${xs(v.pt)}" cy="${y}" r="7" fill="#ff2e88"/><text x="${xs(v.hi)+8}" y="${y+4}" style="fill:#e9e7f5">${f3(v.pt)} [${f3(v.lo)}, ${f3(v.hi)}]</text>`});
  $('#ciPlot').innerHTML=s},'DPD');
 // 03
 const models=[...new Set(D.intersectional.map(r=>r.model))];
 seg($('#interSeg'),models.map(m=>SHORT[m]),sm=>{const m=models.find(x=>SHORT[x]===sm);const rows=D.intersectional.filter(r=>r.model===m);const races=[...new Set(rows.map(r=>r.race))];
  let h=`<div style="display:grid;grid-template-columns:110px repeat(2,1fr);gap:4px"><div></div><div class="hc">Male</div><div class="hc">Female</div>`;
  races.forEach(rc=>{h+=`<div class="hc" style="text-align:left">${rc}</div>`;['Male','Female'].forEach(sx=>{const r=rows.find(q=>q.race===rc&&q.sex===sx);const v=r.sel_rate||0;
   h+=`<div class="hc ${r.status.startsWith('expl')?'exp':''}" style="background:rgba(255,46,136,${.1+v*2})" title="n=${r.n}, positives=${r.n_pos}, TPR=${f3(r.TPR)}">${f3(r.sel_rate)}<br><small>n=${r.n}</small></div>`})});
  const w=D.worst.find(q=>q.model===m);$('#heat').innerHTML=h+`</div><p class="note">Worst-group gap (confirmatory cells): DPD ${f3(w.worst_DPD)} · EOD ${f3(w.worst_EOD)} vs sex-only DPD ${f3(w.sex_only_DPD)}</p>`},'XGBoost');
 // 04
 const P=D.pareto,W=420,H=300,mx=Math.max(...P.map(p=>p.val_DPD))*1.1,ay=[Math.min(...P.map(p=>p.val_acc))-.005,Math.max(...P.map(p=>p.val_acc))+.005];
 const X=v=>40+v/mx*(W-60),Y=v=>H-30-(v-ay[0])/(ay[1]-ay[0])*(H-50);
 let s=`<line x1="${X(.05)}" x2="${X(.05)}" y1="10" y2="${H-30}" stroke="#ffb627" stroke-dasharray="4"/><text x="${X(.05)+4}" y="20" style="fill:#ffb627">DPD ≤ 0.05 rule</text><text x="${W/2-50}" y="${H-6}">validation DPD (sex)</text><text x="2" y="14">val acc</text>`;
 const fr=P.filter(p=>p.pareto).sort((a,b)=>a.val_DPD-b.val_DPD);s+=`<polyline fill="none" stroke="#ff2e88" stroke-dasharray="3" points="${fr.map(p=>X(p.val_DPD)+','+Y(p.val_acc)).join(' ')}"/>`;
 P.forEach((p,i)=>s+=`<circle data-i="${i}" cx="${X(p.val_DPD)}" cy="${Y(p.val_acc)}" r="${p.candidate===D.chosen?9:6}" fill="${p.pareto?'#ff2e88':'#55536b'}" stroke="${p.candidate===D.chosen?'#2ee6d6':'none'}" stroke-width="3" style="cursor:pointer"/>`);
 $('#parSvg').innerHTML=s;
 const showP=i=>{const p=P[i];const t=D.mitigation_test.find(q=>q.model.endsWith(p.candidate));$('#parInfo').innerHTML=`<b>${p.candidate}</b>${p.candidate===D.chosen?' <span style="color:#2ee6d6">◆ chosen by prespecified rule</span>':''}<br>validation accuracy ${f3(p.val_acc)} · DPD ${f3(p.val_DPD)} · ${p.pareto?'on Pareto front':'dominated'}`+(t?`<br>test: accuracy ${f3(t.accuracy)} · F1 ${f3(t.f1)} · DPD ${f3(t.DPD)} · EOD ${f3(t.EOD)} · FPRD ${f3(t.FPRD)}`:'<br><span class="note">Only the chosen point (and the baseline) are evaluated on test.</span>')};
 $('#parSvg').onclick=e=>{if(e.target.dataset.i)showP(+e.target.dataset.i)};showP(P.findIndex(p=>p.candidate===D.chosen));
 // 05
 table($('#t6'),[['method','Method'],['consistency_spearman','Consistency ρ'],['consistency_top10_jaccard','Top-10 Jaccard'],['stability_spearman','Stability ρ'],['stability_top10_jaccard','Stab. top-10 J'],['fidelity_deletion_meanAUCdrop','Deletion ΔAUC'],['fidelity_vs_random_ratio','× random'],['fidelity_spearman_vs_single_deletion','Fidelity ρ']],D.table6,['consistency_spearman','consistency_top10_jaccard','stability_spearman','fidelity_spearman_vs_single_deletion']);
 // 06
 $('#govList').innerHTML=D.governance.map(g=>`<div class="gv"><span>${g.construct}</span><div class="bar"><div style="width:${g.jaccard*100}%;background:${g.jaccard<1?'linear-gradient(90deg,#ffb627,#ff2e88)':'linear-gradient(90deg,#8c7bff,#2ee6d6)'}"></div></div><b>${f3(g.jaccard)}</b></div>`).join('');
 // 08
 table($('#pv'),[['metric','Metric'],['paper','Paper'],['reproduced','Reproduced'],['difference','Δ']],D.paper_vs);
 [...$('#pv').querySelectorAll('tr')].slice(1).forEach(tr=>{const td=tr.lastChild;td.className=parseFloat(td.textContent)<0?'neg':'pos'});
 // 07 what-if
 const L=D.lr,F=$('#wf'),defs={age:38,fnlwgt:178000,'education-num':10,'capital-gain':0,'capital-loss':0,'hours-per-week':40};
 L.num.forEach(n=>F.insertAdjacentHTML('beforeend',`<label>${n}<input type="number" name="${n}" value="${defs[n]}"></label>`));
 Object.entries(L.cats).forEach(([c,o])=>F.insertAdjacentHTML('beforeend',`<label>${c}<select name="${c}">${o.levels.map(l=>`<option${l===o.impute?' selected':''}>${l}</option>`).join('')}</select></label>`));
 const run=()=>{let z=L.intercept;L.num.forEach((n,i)=>{const v=parseFloat(F[n].value);z+=L.num_coef[i]*(((isNaN(v)?L.num_impute[i]:v)-L.mean[i])/L.scale[i])});
  Object.entries(L.cats).forEach(([c,o])=>{const j=o.levels.indexOf(F[c].value);if(j>=0)z+=o.coef[j]});
  const p=1/(1+Math.exp(-z)),sx=F['sex'].value,g=$('#gt').checked,t=g?L.group.thresholds[sx]:L.threshold;
  $('#gfill').style.width=(p*100)+'%';$('#wout').innerHTML=`P(income &gt; 50K) = <b>${f3(p)}</b> · threshold ${f3(t)} ${g?`(${sx})`:'(single)'} → <b style="color:${p>=t?'#2ee6d6':'#ff2e88'}">${p>=t?'>50K':'≤50K'}</b>`;
  const m=g?L.group.test:L.base;$('#wcmp').textContent=`Logistic Regression on test with ${g?'group':'single'} thresholds: accuracy ${f3(m.accuracy)} · DPD ${f3(m.DPD)} · EOD ${f3(m.EOD)} · FPRD ${f3(m.FPRD)}. Illustration only, not for decisions about individuals.`};
 F.oninput=run;$('#gt').onchange=run;run();
});
