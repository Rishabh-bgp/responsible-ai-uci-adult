"""Responsible AI framework on UCI Adult: provenance, governance crosswalk, models, fairness, mitigation,
explanation audit and reproducibility log. Plain-Python version of notebooks/Responsible_AI_UCI_Adult.ipynb.

Run:  python scripts/responsible_ai_uci_adult.py   (writes everything to results/)
"""
import os, sys
RESULTS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
def display(x): print(x)

def setup():
    global CONFIG, OUT, SEED, T0, datetime, hashlib, json, matplotlib, np, os, pd, platform, plt, random, time, urllib, warnings
    import os, json, time, hashlib, random, platform, warnings, urllib.request, datetime
    import numpy as np, pandas as pd, matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    warnings.filterwarnings('ignore')
    SEED=42; np.random.seed(SEED); random.seed(SEED); os.environ['PYTHONHASHSEED']=str(SEED)
    OUT='.'; T0=time.time()
    CONFIG=dict(seed=SEED, split=dict(train=0.6,val=0.2,test=0.2,stratify='income x sex'),
                n_boot=5000, min_cell=50, min_pos=10, cv_folds=3, threshold_rule='max F1 on validation',
                pareto_rule='max validation accuracy s.t. validation DPD(sex) <= 0.05; fallback min DPD',
                protected=dict(sex={'Male':'Group A','Female':'Group B'}, race='as in source'))

def provenance():
    global BASE, cols, df, f, files, k, miss, p, provenance, retrieved, sha, te, tr, v
    BASE='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
    files=['adult.data','adult.test','adult.names']; os.makedirs('data',exist_ok=True); sha={}
    retrieved=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for f in files:
        p=f'data/{f}'
        if not os.path.exists(p): urllib.request.urlretrieve(BASE+f,p)
        sha[f]=hashlib.sha256(open(p,'rb').read()).hexdigest()
    cols=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship',
          'race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
    tr=pd.read_csv('data/adult.data',names=cols,skipinitialspace=True,na_values='?')
    te=pd.read_csv('data/adult.test',names=cols,skipinitialspace=True,na_values='?',skiprows=1)
    df=pd.concat([tr,te],ignore_index=True); df['income']=df['income'].str.rstrip('.')
    df['y']=(df['income']=='>50K').astype(int)
    miss=df[cols].isna().sum(); miss=miss[miss>0]
    provenance=dict(title='Adult (Census Income)',doi='10.24432/C5XW20',license='CC BY 4.0',
      citation='Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20',
      source_url=BASE,retrieved_utc=retrieved,sha256=sha,n_instances=int(len(df)),n_features=14,
      positive_rate=float(df.y.mean()),
      missing_values={k:dict(count=int(v),pct=round(100*v/len(df),3)) for k,v in miss.items()},
      rows_with_any_missing=int(df[cols].isna().any(axis=1).sum()),
      quasi_identifiers=['age','sex','race','native-country','marital-status','relationship','education','occupation'],
      mitigations=['no re-identification attempts','no external linkage','access-controlled storage','aggregate-only reporting (no row-level outputs)'],
      purpose_limitation='research benchmark for fairness/XAI auditing only; not for decisions about individuals',
      legal_basis='public dataset under CC BY 4.0; secondary research use',
      temporal_bias='1994 Census (CPS) extract; labour-market conditions and income threshold are dated',
      sampling_bias='U.S. population only; fnlwgt weighting not used in modelling; race categories coarse',
      label_notes='>50K label reflects historical labour-market inequality')
    json.dump(provenance,open('provenance.json','w'),indent=2); print(json.dumps(provenance,indent=2))

def governance_crosswalk():
    global A, E, M, P, R, c, crosswalk, d, gov, jacc, k, v
    P,E,M,R='procedure','evidence','monitoring','risk_tier'
    A={P,E,M,R}
    crosswalk={
     'Accountability':dict(NIST=A,OECD=A,EU_AI_Act=A,IEEE7000=A),
     'Transparency & explainability':dict(NIST=A,OECD=A,EU_AI_Act=A,IEEE7000=A),
     'Fairness / non-discrimination':dict(NIST=A,OECD=A,EU_AI_Act=A,IEEE7000=A),
     'Human oversight':dict(NIST=A,OECD=A,EU_AI_Act=A,IEEE7000=A),
     'Robustness, safety & security':dict(NIST=A,OECD=A,EU_AI_Act=A,IEEE7000=A),
     'Purpose limitation':dict(NIST={P,E},OECD={P,E},EU_AI_Act={P,E,M},IEEE7000={P,E,M}),
     'Data protection impact assessment':dict(NIST={P,E,M},OECD={P,E,M},EU_AI_Act=A,IEEE7000={P,E,M}),
     'Independent review / audit':dict(NIST={P,E,M},OECD={P,E,M},EU_AI_Act=A,IEEE7000={P,E,M}),
    }
    def jacc(sets): return len(set.intersection(*sets))/len(set.union(*sets))
    gov=pd.DataFrame([{'construct':c,**{k:','.join(sorted(v)) for k,v in d.items()},'jaccard':round(jacc(list(d.values())),3)} for c,d in crosswalk.items()])
    gov.to_csv('governance_jaccard.csv',index=False); print(gov)

def nist_counts():
    global cats, counts, fn, i, j, n, nist_ids, nist_struct, nist_tab, paper_counts, s
    # NIST AI RMF 1.0 subcategory catalogue (IDs as in the published Core)
    nist_struct={'GOVERN':[7,3,2,3,2,2],'MAP':[6,3,5,2,2],'MEASURE':[3,13,3,3],'MANAGE':[4,4,2,3]}
    nist_ids=[f'{fn} {i+1}.{j+1}' for fn,cats in nist_struct.items() for i,n in enumerate(cats) for j in range(n)]
    counts=pd.Series([s.split()[0] for s in nist_ids]).value_counts().reindex(nist_struct.keys())
    paper_counts={'GOVERN':19,'MAP':18,'MEASURE':21,'MANAGE':14}
    nist_tab=pd.DataFrame({'paper':paper_counts,'encoded_from_AI_RMF_1.0_Core':counts}); nist_tab.loc['TOTAL']=nist_tab.sum()
    print('Note: the published Core lists MEASURE 22 / MANAGE 13; the paper reports 21 / 14. Both total 72.'); print(nist_tab)

def control_catalog():
    global artifacts, cadence_by_fn, catalog, etype, k, s
    artifacts=['risk register','model card','datasheet/provenance JSON','DPIA','fairness audit report','explanation audit report','monitoring dashboard','incident log']
    cadence_by_fn={'GOVERN':'Annual','MAP':'Per release','MEASURE':'Quarterly','MANAGE':'Continuous / event-triggered'}
    etype=['documented procedure','audit evidence','operational monitoring']
    catalog=pd.DataFrame([dict(control_id=f'CTL-{k+1:03d}',nist_subcategory=s,function=s.split()[0],
       required_artifact=artifacts[k%8],evidence_type=etype[k%3],review_cadence=cadence_by_fn[s.split()[0]]) for k,s in enumerate(nist_ids)])
    catalog.to_csv('control_artifact_catalog.csv',index=False)
    print('Table 1/2-style sample (generic illustrative mapping; full 72-row catalogue saved to CSV):')
    print(catalog.groupby('function').head(2))

def train_models():
    global ColumnTransformer, GridSearchCV, LogisticRegression, MLPClassifier, OneHotEncoder, Pipeline, RandomForestClassifier, SimpleImputer, StandardScaler, StratifiedKFold, X, XGBClassifier, Xte, Xtmp, Xtr, Xva, accuracy_score, best_params, cat, cv, f1_score, g, gs, m, models, n, num, precision_score, prep, recall_score, roc_auc_score, s_tmp, s_tr, specs, strat, t, train_test_split, y, yte, ytmp, ytr, yva
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
    num=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    cat=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    X=df[num+cat]; y=df.y.values; strat=df.y.astype(str)+df.sex
    Xtr,Xtmp,ytr,ytmp,s_tr,s_tmp=train_test_split(X,y,strat,test_size=0.4,stratify=strat,random_state=SEED)
    Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=0.5,stratify=s_tmp,random_state=SEED)
    print(len(Xtr),len(Xva),len(Xte))
    def prep(): return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),
       ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    specs={'Logistic Regression':(LogisticRegression(max_iter=2000,random_state=SEED),{'clf__C':[0.1,1,10]}),
     'Random Forest':(RandomForestClassifier(random_state=SEED,n_jobs=-1),{'clf__n_estimators':[300],'clf__max_depth':[12,20],'clf__min_samples_leaf':[1,3]}),
     'XGBoost':(XGBClassifier(random_state=SEED,n_jobs=-1,eval_metric='logloss',tree_method='hist'),{'clf__n_estimators':[300,600],'clf__max_depth':[4,6],'clf__learning_rate':[0.05]}),
     'Neural Network (MLP)':(MLPClassifier(random_state=SEED,max_iter=300,early_stopping=True),{'clf__hidden_layer_sizes':[(64,),(64,32)],'clf__alpha':[1e-4,1e-2]})}
    cv=StratifiedKFold(3,shuffle=True,random_state=SEED); models={}; best_params={}
    for n,(m,g) in specs.items():
        t=time.time(); gs=GridSearchCV(Pipeline([('prep',prep()),('clf',m)]),g,cv=cv,scoring='roc_auc',n_jobs=1 if n!='Logistic Regression' else -1).fit(Xtr,ytr)
        models[n]=gs.best_estimator_; best_params[n]=gs.best_params_; print(n,gs.best_params_,round(gs.best_score_,4),f'{time.time()-t:.0f}s')

def thresholds_and_table3():
    global P_te, P_va, best_thr, m, n, rows, t, table3, thr, ts, yh
    def best_thr(p,y):
        ts=np.linspace(0.05,0.95,181); return float(ts[np.argmax([f1_score(y,p>=t) for t in ts])])
    thr={}; P_va={}; P_te={}; rows=[]
    for n,m in models.items():
        P_va[n]=m.predict_proba(Xva)[:,1]; P_te[n]=m.predict_proba(Xte)[:,1]; thr[n]=best_thr(P_va[n],yva)
    for n in models:   # test set touched only after all thresholds frozen
        yh=(P_te[n]>=thr[n]).astype(int)
        rows.append(dict(model=n,threshold=round(thr[n],3),accuracy=accuracy_score(yte,yh),precision=precision_score(yte,yh),
          recall=recall_score(yte,yh),f1=f1_score(yte,yh),roc_auc=roc_auc_score(yte,P_te[n])))
    table3=pd.DataFrame(rows).set_index('model').round(3); table3.to_csv('table3_performance.csv'); print(table3)

def table4_group_rates():
    global c, confusion_matrix, fn, fp, g, race_te, rates, s, sex_te, t4, tn, tp, yh_x
    from sklearn.metrics import confusion_matrix
    sex_te=Xte['sex'].values; race_te=Xte['race'].values
    def rates(y,yh): tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel(); return dict(n=len(y),TN=tn,FP=fp,FN=fn,TP=tp,PPR=(tp+fp)/len(y),TPR=tp/(tp+fn) if tp+fn else np.nan,FPR=fp/(fp+tn) if fp+tn else np.nan)
    yh_x=(P_te['XGBoost']>=thr['XGBoost']).astype(int)
    t4=pd.DataFrame({g:rates(yte[sex_te==s],yh_x[sex_te==s]) for s,g in [('Male','Group A (Male)'),('Female','Group B (Female)')]}).T
    t4.loc['|A-B|']=np.nan
    for c in ['PPR','TPR','FPR']: t4.loc['|A-B|',c]=abs(t4.iloc[0][c]-t4.iloc[1][c])
    t4.to_csv('table4_group_rates_xgb.csv'); print(t4)

def table5_bootstrap_fairness():
    global B, _, a, b, boot_idx, bs, c, f, gaps, hi, i, ix, k, lo, n, out, pt, rng, rows, s, strata, t5, yh
    def gaps(y,yh,g):
        out=[]
        for f in (lambda m:yh[m].mean(), lambda m:yh[m&(y==1)].mean(), lambda m:yh[m&(y==0)].mean()):
            a,b=f(g=='Male'),f(g=='Female'); out.append(abs(a-b))
        return out
    rng=np.random.default_rng(SEED); B=CONFIG['n_boot']
    strata=[np.where((sex_te==s)&(yte==c))[0] for s in ['Male','Female'] for c in [0,1]]
    boot_idx=[np.concatenate([rng.choice(ix,len(ix)) for ix in strata]) for _ in range(B)]
    rows=[]
    for n in models:
        yh=(P_te[n]>=thr[n]).astype(int); pt=gaps(yte,yh,sex_te)
        bs=np.array([gaps(yte[ix],yh[ix],sex_te[ix]) for ix in boot_idx]); lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
        rows.append({'model':n,**{f'{k}':f'{pt[i]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]' for i,k in enumerate(['DPD','EOD','FPRD'])},'_pt':pt,'_lo':lo,'_hi':hi})
    t5=pd.DataFrame(rows).set_index('model'); t5[['DPD','EOD','FPRD']].to_csv('table5_fairness_ci.csv'); print(f'B={B} stratified bootstrap resamples'); print(t5[['DPD','EOD','FPRD']])

def plot_fairness_ci():
    global ax, fig, hi, i, k, lo, pt, r, w, xs
    fig,ax=plt.subplots(figsize=(8,4)); w=0.25; xs=np.arange(len(t5))
    for i,k in enumerate(['DPD','EOD','FPRD']):
        pt=np.array([r[i] for r in t5._pt]); lo=np.array([r[i] for r in t5._lo]); hi=np.array([r[i] for r in t5._hi])
        ax.bar(xs+i*w,pt,w,yerr=[pt-lo,hi-pt],capsize=3,label=k)
    ax.set_xticks(xs+w); ax.set_xticklabels(t5.index,rotation=10); ax.set_ylabel('absolute gap (sex)'); ax.legend(); ax.set_title('Fairness gaps with 95% stratified bootstrap CI')
    plt.tight_layout(); plt.savefig('figures/fig_fairness_ci.png',dpi=130); plt.show()

def intersectional_audit():
    global MIN, MINP, conf, inter, m, n, npos, r, s, worst, yh
    MIN=CONFIG['min_cell']; MINP=CONFIG['min_pos']; inter=[]
    for n in models:
        yh=(P_te[n]>=thr[n]).astype(int)
        for s in ['Male','Female']:
            for r in sorted(set(race_te)):
                m=(sex_te==s)&(race_te==r); npos=int(yte[m].sum())
                inter.append(dict(model=n,sex=s,race=r,n=int(m.sum()),n_pos=npos,sel_rate=yh[m].mean() if m.sum() else np.nan,
                  TPR=yh[m&(yte==1)].mean() if npos>0 else np.nan, status='confirmatory' if m.sum()>=MIN and npos>=MINP else 'exploratory (sparse)'))
    inter=pd.DataFrame(inter); inter.to_csv('intersectional_sex_race.csv',index=False)
    conf=inter[inter.status=='confirmatory']
    worst=conf.groupby('model').agg(worst_DPD=('sel_rate',lambda s:s.max()-s.min()),worst_EOD=('TPR',lambda s:s.max()-s.min()))
    worst['sex_only_DPD']=[t5.loc[n,'_pt'][0] for n in worst.index]
    print(inter[inter.model=='XGBoost'].round(3)); print(worst.round(3))

def plot_intersectional():
    global ax, cols_, fig, px, s
    px=inter[inter.model=='XGBoost'].copy(); px['cell']=px.sex.str[0]+' | '+px.race
    fig,ax=plt.subplots(figsize=(9,4)); cols_=['tab:blue' if s=='confirmatory' else 'lightgray' for s in px.status]
    ax.bar(px.cell,px.sel_rate,color=cols_); ax.set_ylabel('selection rate'); ax.set_title('XGBoost selection rate by sex x race (grey = exploratory, sparse)')
    plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.savefig('figures/fig_intersectional.png',dpi=130); plt.show()

def mitigation_candidates():
    global BaseEstimator, ClassifierMixin, DemographicParity, ExponentiatedGradient, F64, ThresholdOptimizer, Zte, Ztr, Zva, __init__, __sklearn_is_fitted__, c, cands, chosen, clone, cons, dpd, eg, eps, fit, k, n, o, ok, pg, pre, predict, predict_proba, prw, pv, r, rw, s, sex_tr, sex_va, to, trw, v, w, xgb64
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from fairlearn.postprocessing import ThresholdOptimizer
    from sklearn.base import clone
    sex_tr=Xtr['sex'].values; sex_va=Xva['sex'].values
    def dpd(yh,g): return abs(yh[g=='Male'].mean()-yh[g=='Female'].mean())
    cands={}
    for n in models: cands[n+' (baseline)']=dict(val=(P_va[n]>=thr[n]).astype(int),test_fn=lambda n=n:(P_te[n]>=thr[n]).astype(int))
    pg=pd.crosstab(sex_tr,ytr,normalize=True); w=np.array([pg.loc[s].sum()*pg[c].sum()/pg.loc[s,c] for s,c in zip(sex_tr,ytr)])
    rw=clone(models['XGBoost']).fit(Xtr,ytr,clf__sample_weight=w); prw=rw.predict_proba(Xva)[:,1]; trw=best_thr(prw,yva)
    cands['XGBoost + reweighing']=dict(val=(prw>=trw).astype(int),test_fn=lambda:(rw.predict_proba(Xte)[:,1]>=trw).astype(int))
    pre=prep().fit(Xtr); Ztr,Zva,Zte=pre.transform(Xtr),pre.transform(Xva),pre.transform(Xte)
    for eps in [0.01,0.02,0.05,0.1]:
        eg=ExponentiatedGradient(LogisticRegression(max_iter=2000,C=best_params['Logistic Regression']['clf__C']),DemographicParity(difference_bound=eps),eps=eps)
        np.random.seed(SEED); eg.fit(Ztr,ytr,sensitive_features=sex_tr)
        cands[f'ExpGrad-LR eps={eps}']=dict(val=eg.predict(Zva,random_state=SEED),test_fn=lambda eg=eg:eg.predict(Zte,random_state=SEED))
    from sklearn.base import BaseEstimator, ClassifierMixin
    class F64(BaseEstimator,ClassifierMixin):   # float64 wrapper: avoids a fairlearn/pandas float32 setitem error
        def __init__(self,m): self.m=m
        def fit(self,X,y): return self
        def predict_proba(self,X): return self.m.predict_proba(X).astype('float64')
        def predict(self,X): return self.m.predict(X)
        def __sklearn_is_fitted__(self): return True
    xgb64=F64(models['XGBoost']); xgb64.classes_=np.array([0,1])
    for cons in ['demographic_parity','equalized_odds','true_positive_rate_parity']:
        to=ThresholdOptimizer(estimator=xgb64,constraints=cons,objective='accuracy_score',prefit=True,predict_method='predict_proba')
        to.fit(Xva,yva,sensitive_features=sex_va)
        cands[f'XGBoost + ThresholdOpt ({cons})']=dict(val=to.predict(Xva,sensitive_features=sex_va,random_state=SEED),test_fn=lambda to=to:to.predict(Xte,sensitive_features=sex_te,random_state=SEED))
    pv=pd.DataFrame([dict(candidate=k,val_acc=accuracy_score(yva,v['val']),val_DPD=dpd(v['val'],sex_va)) for k,v in cands.items()])
    pv['pareto']=[not any((o.val_acc>=r.val_acc)&(o.val_DPD<=r.val_DPD)&((o.val_acc>r.val_acc)|(o.val_DPD<r.val_DPD)) for o in pv.itertuples()) for r in pv.itertuples()]
    ok=pv[pv.val_DPD<=0.05]; chosen=ok.sort_values('val_acc').iloc[-1].candidate if len(ok) else pv.sort_values('val_DPD').iloc[0].candidate
    print(pv.round(4))

def pareto_and_test():
    global ax, f, fig, k, mit, p, r, v, yb, yc
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(pv.val_DPD,pv.val_acc,c=['tab:red' if p else 'gray' for p in pv.pareto])
    f=pv[pv.pareto].sort_values('val_DPD'); ax.plot(f.val_DPD,f.val_acc,'r--')
    for r in pv.itertuples(): ax.annotate(r.candidate.replace('XGBoost','XGB'),(r.val_DPD,r.val_acc),fontsize=6)
    ax.axvline(0.05,ls=':'); ax.set_xlabel('validation DPD (sex)'); ax.set_ylabel('validation accuracy'); ax.set_title('Validation fairness-accuracy Pareto front')
    plt.tight_layout(); plt.savefig('figures/fig_pareto.png',dpi=130); plt.show()
    yc=cands[chosen]['test_fn'](); yb=(P_te['XGBoost']>=thr['XGBoost']).astype(int)
    mit=pd.DataFrame({k:dict(accuracy=accuracy_score(yte,v),f1=f1_score(yte,v),DPD=gaps(yte,v,sex_te)[0],EOD=gaps(yte,v,sex_te)[1],FPRD=gaps(yte,v,sex_te)[2]) for k,v in [('XGBoost baseline',yb),('Chosen: '+chosen,yc)]}).T.round(3)
    mit.to_csv('mitigation_test.csv'); print('Chosen by prespecified rule:',chosen); print(mit)

def explanation_runs():
    global Zte_x, Ztr_x, Zva_x, bg, cons, d, e, ev, fn, i, imp, j, lime, lime_imp, lime_ix, permutation_importance, pert, pi_imp, pr, ps, r0, runs, s, seeds, shap, shap_imp, spearmanr, t, top, v, xclf, xpre
    import shap, lime.lime_tabular
    from scipy.stats import spearmanr
    from sklearn.inspection import permutation_importance
    xpre=models['XGBoost'].named_steps['prep']; xclf=models['XGBoost'].named_steps['clf']
    fn=list(xpre.get_feature_names_out()); Ztr_x,Zva_x,Zte_x=[xpre.transform(d) for d in (Xtr,Xva,Xte)]
    r0=np.random.default_rng(SEED); ev=Zva_x[r0.choice(len(Zva_x),300,replace=False)]; lime_ix=r0.choice(len(Zva_x),100,replace=False)
    pr=lambda Z: xclf.predict_proba(Z)
    def shap_imp(seed,bg_src=Ztr_x,nbg=200):
        bg=bg_src[np.random.default_rng(seed).choice(len(bg_src),nbg,replace=False)]
        return np.abs(shap.TreeExplainer(xclf,data=bg,feature_perturbation='interventional').shap_values(ev)).mean(0)
    def lime_imp(seed,kw=None,ns=1000):
        e=lime.lime_tabular.LimeTabularExplainer(Ztr_x,feature_names=fn,discretize_continuous=False,random_state=seed,kernel_width=kw)
        imp=np.zeros(len(fn))
        for i in lime_ix:
            for j,v in e.explain_instance(Zva_x[i],pr,num_features=20,num_samples=ns).as_map()[1]: imp[j]+=abs(v)
        return imp/len(lime_ix)
    def pi_imp(seed,Z=Zva_x,yy=yva):
        return permutation_importance(xclf,Z,yy,n_repeats=5,random_state=seed,scoring='roc_auc',n_jobs=-1).importances_mean
    def top(a,k=10): return set(np.argsort(-a)[:k])
    def cons(runs):
        ps=[(i,j) for i in range(len(runs)) for j in range(i+1,len(runs))]
        return np.mean([spearmanr(runs[i],runs[j])[0] for i,j in ps]), np.mean([len(top(runs[i])&top(runs[j]))/len(top(runs[i])|top(runs[j])) for i,j in ps])
    seeds=[0,1,2,3,4]; t=time.time()
    runs={'SHAP':[shap_imp(s) for s in seeds]}; print('shap',time.time()-t)
    runs['LIME']=[lime_imp(s) for s in seeds]; print('lime',time.time()-t)
    runs['Permutation importance']=[pi_imp(s) for s in seeds]; print('pi',time.time()-t)
    pert={'SHAP':shap_imp(99,Zte_x,50),'LIME':lime_imp(99,kw=0.5*np.sqrt(Ztr_x.shape[1])*0.75,ns=500),'Permutation importance':pi_imp(99,Zte_x,yte)}

def explanation_metrics():
    global R, Z, Zd, base_auc, c_jac, c_rho, dc, del_curve, dz, j, m, mu, out, rand, ref, rows, s, single, st, t6, yd
    dz=r0.choice(len(Zva_x),2000,replace=False); Zd,yd=Zva_x[dz],yva[dz]; mu=Ztr_x.mean(0)
    base_auc=roc_auc_score(yd,pr(Zd)[:,1])
    def del_curve(order,K=10):
        Z=Zd.copy(); out=[]
        for j in order[:K]: Z[:,j]=mu[j]; out.append(base_auc-roc_auc_score(yd,pr(Z)[:,1]))
        return np.array(out)
    single=np.array([base_auc-roc_auc_score(yd,pr(np.where(np.arange(Zd.shape[1])==j,mu,Zd))[:,1]) for j in range(Zd.shape[1])])
    rand=np.mean([del_curve(np.random.default_rng(s).permutation(len(fn))).mean() for s in range(10)])
    rows=[]
    for m,R in runs.items():
        ref=np.mean(R,0); c_rho,c_jac=cons(R); st=spearmanr(R[0],pert[m])[0]
        dc=del_curve(np.argsort(-ref))
        rows.append(dict(method=m,consistency_spearman=c_rho,consistency_top10_jaccard=c_jac,stability_spearman=st,
           stability_top10_jaccard=len(top(R[0])&top(pert[m]))/len(top(R[0])|top(pert[m])),
           fidelity_deletion_meanAUCdrop=dc.mean(),fidelity_vs_random_ratio=dc.mean()/rand,fidelity_spearman_vs_single_deletion=spearmanr(ref,single)[0]))
    t6=pd.DataFrame(rows).set_index('method').round(3); t6.to_csv('table6_explanations.csv')
    print('random-order mean AUC drop:',round(rand,4)); print(t6)

def plot_shap():
    global ax, ref
    ref=pd.Series(np.mean(runs['SHAP'],0),index=fn).sort_values()[-15:]
    ax=ref.plot.barh(figsize=(7,5),title='Frozen XGBoost: mean |SHAP| (top 15)'); plt.tight_layout(); plt.savefig('figures/fig_shap_top15.png',dpi=130); plt.show()

def reproducibility_log():
    global _l, _s, a, b, fairlearn, k, p, repro, scipy, sklearn, v, version, xgboost
    import sklearn, xgboost, shap as _s, lime as _l, fairlearn, scipy
    from importlib.metadata import version
    repro=dict(config=CONFIG,seeds=dict(global_=SEED,explainer_runs=seeds,bootstrap=SEED),best_params={k:{a:str(b) for a,b in v.items()} for k,v in best_params.items()},
      thresholds=thr,split_sizes=dict(train=len(Xtr),val=len(Xva),test=len(Xte)),data_sha256=sha,
      packages={p:version(p) for p in ['numpy','pandas','scikit-learn','xgboost','shap','lime','fairlearn','scipy','matplotlib']},
      python=platform.python_version(),platform=platform.platform(),runtime_sec=round(time.time()-T0,1),chosen_mitigation=chosen)
    json.dump(repro,open('reproducibility_log.json','w'),indent=2,default=str); print(json.dumps(repro,indent=2,default=str))

def paper_vs_reproduced():
    global X3, cmp, f5, gj, got, paper
    paper={'XGBoost acc':0.901,'XGBoost F1':0.862,'XGBoost AUC':0.951,'MLP acc':0.893,'MLP F1':0.851,'RF acc':0.879,'RF F1':0.833,'LR acc':0.842,'LR AUC':0.891,
     'XGB DPD':0.162,'XGB EOD':0.128,'XGB FPRD':0.104,'MLP DPD':0.151,'MLP EOD':0.119,'MLP FPRD':0.096,'LR DPD':0.102,'LR EOD':0.081,'LR FPRD':0.063,
     'XGB worst-group intersectional DPD':0.214,'SHAP consistency rho':0.91,'SHAP top10 Jaccard':0.88,'PI consistency rho':0.87,'LIME consistency rho':0.82,
     'PI top10 Jaccard':0.83,'LIME top10 Jaccard':0.74,'SHAP stability':0.86,'LIME stability':0.71,'PI stability':0.80,'PI fidelity':0.88,'SHAP fidelity':0.84,'LIME fidelity':0.79}
    X3=table3; f5=lambda n,i:t5.loc[n,'_pt'][i]
    got={'XGBoost acc':X3.loc['XGBoost','accuracy'],'XGBoost F1':X3.loc['XGBoost','f1'],'XGBoost AUC':X3.loc['XGBoost','roc_auc'],
     'MLP acc':X3.loc['Neural Network (MLP)','accuracy'],'MLP F1':X3.loc['Neural Network (MLP)','f1'],'RF acc':X3.loc['Random Forest','accuracy'],'RF F1':X3.loc['Random Forest','f1'],
     'LR acc':X3.loc['Logistic Regression','accuracy'],'LR AUC':X3.loc['Logistic Regression','roc_auc'],
     'XGB DPD':f5('XGBoost',0),'XGB EOD':f5('XGBoost',1),'XGB FPRD':f5('XGBoost',2),'MLP DPD':f5('Neural Network (MLP)',0),'MLP EOD':f5('Neural Network (MLP)',1),'MLP FPRD':f5('Neural Network (MLP)',2),
     'LR DPD':f5('Logistic Regression',0),'LR EOD':f5('Logistic Regression',1),'LR FPRD':f5('Logistic Regression',2),
     'XGB worst-group intersectional DPD':worst.loc['XGBoost','worst_DPD'],
     'SHAP consistency rho':t6.loc['SHAP','consistency_spearman'],'SHAP top10 Jaccard':t6.loc['SHAP','consistency_top10_jaccard'],
     'PI consistency rho':t6.loc['Permutation importance','consistency_spearman'],'LIME consistency rho':t6.loc['LIME','consistency_spearman'],
     'PI top10 Jaccard':t6.loc['Permutation importance','consistency_top10_jaccard'],'LIME top10 Jaccard':t6.loc['LIME','consistency_top10_jaccard'],
     'SHAP stability':t6.loc['SHAP','stability_spearman'],'LIME stability':t6.loc['LIME','stability_spearman'],'PI stability':t6.loc['Permutation importance','stability_spearman'],
     'PI fidelity':t6.loc['Permutation importance','fidelity_spearman_vs_single_deletion'],'SHAP fidelity':t6.loc['SHAP','fidelity_spearman_vs_single_deletion'],'LIME fidelity':t6.loc['LIME','fidelity_spearman_vs_single_deletion']}
    cmp=pd.DataFrame({'paper':paper,'reproduced':got}); cmp['difference']=cmp.reproduced-cmp.paper
    gj=dict(zip(gov.construct,gov.jaccard)); print('Governance Jaccard (follows from the interpretive coding):',gj)
    cmp.round(3).to_csv('paper_vs_reproduced.csv'); print(cmp.round(3))

def export_web_results():
    global W, c, cats, coef, cs, gthr, i, imc, imn, k, lc, lp, lr, lr_base, lr_group, m, n, oh, pl, pos, ps, s_, sc, target, v, ylg
    lr=models['Logistic Regression']; lp=lr.named_steps['prep']; lc=lr.named_steps['clf']
    sc=lp.named_transformers_['num'].named_steps['sc']; imn=lp.named_transformers_['num'].named_steps['imp']
    oh=lp.named_transformers_['cat'].named_steps['oh']; imc=lp.named_transformers_['cat'].named_steps['imp']
    coef=lc.coef_[0]; k=len(num); cats={}; pos=k
    for c,cs in zip(cat,oh.categories_): cats[c]={'levels':[str(v) for v in cs],'coef':coef[pos:pos+len(cs)].tolist(),'impute':str(imc.statistics_[cat.index(c)])}; pos+=len(cs)
    pl=P_va['Logistic Regression']; target=(pl>=thr['Logistic Regression']).mean(); gthr={}
    for s_ in ['Male','Female']:
        ps=np.sort(pl[sex_va==s_])[::-1]; gthr[s_]=float(ps[int(round(target*len(ps)))-1])
    ylg=np.where(sex_te=='Male',P_te['Logistic Regression']>=gthr['Male'],P_te['Logistic Regression']>=gthr['Female']).astype(int)
    lr_group=dict(thresholds=gthr,test=dict(accuracy=accuracy_score(yte,ylg),f1=f1_score(yte,ylg),DPD=gaps(yte,ylg,sex_te)[0],EOD=gaps(yte,ylg,sex_te)[1],FPRD=gaps(yte,ylg,sex_te)[2]))
    lr_base=dict(accuracy=table3.loc['Logistic Regression','accuracy'],DPD=t5.loc['Logistic Regression','_pt'][0],EOD=t5.loc['Logistic Regression','_pt'][1],FPRD=t5.loc['Logistic Regression','_pt'][2])
    print(lr_group)
    W=dict(n=int(len(df)),split=dict(train=len(Xtr),val=len(Xva),test=len(Xte)),n_boot=B,
     table3=table3.reset_index().to_dict('records'),
     table4=t4.reset_index().rename(columns={'index':'group'}).astype(object).where(t4.reset_index().notna(),None).to_dict('records'),
     table5=[dict(model=n,**{m:dict(pt=float(t5.loc[n,'_pt'][i]),lo=float(t5.loc[n,'_lo'][i]),hi=float(t5.loc[n,'_hi'][i])) for i,m in enumerate(['DPD','EOD','FPRD'])}) for n in t5.index],
     intersectional=inter.astype(object).where(inter.notna(),None).to_dict('records'),worst=worst.reset_index().to_dict('records'),
     pareto=pv.to_dict('records'),chosen=chosen,mitigation_test=mit.reset_index().rename(columns={'index':'model'}).to_dict('records'),
     table6=t6.reset_index().to_dict('records'),governance=gov.to_dict('records'),
     paper_vs=cmp.reset_index().rename(columns={'index':'metric'}).to_dict('records'),
     lr=dict(num=num,mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),num_impute=imn.statistics_.tolist(),num_coef=coef[:k].tolist(),cats=cats,
       intercept=float(lc.intercept_[0]),threshold=thr['Logistic Regression'],group=lr_group,base=lr_base))
    json.dump(W,open('web_results.json','w'),default=float,indent=1); print('saved web_results.json')

def main():
    os.makedirs(os.path.join(RESULTS,'figures'),exist_ok=True); os.chdir(RESULTS)
    print("\n=== setup ==="); setup()
    print("\n=== provenance ==="); provenance()
    print("\n=== governance_crosswalk ==="); governance_crosswalk()
    print("\n=== nist_counts ==="); nist_counts()
    print("\n=== control_catalog ==="); control_catalog()
    print("\n=== train_models ==="); train_models()
    print("\n=== thresholds_and_table3 ==="); thresholds_and_table3()
    print("\n=== table4_group_rates ==="); table4_group_rates()
    print("\n=== table5_bootstrap_fairness ==="); table5_bootstrap_fairness()
    print("\n=== plot_fairness_ci ==="); plot_fairness_ci()
    print("\n=== intersectional_audit ==="); intersectional_audit()
    print("\n=== plot_intersectional ==="); plot_intersectional()
    print("\n=== mitigation_candidates ==="); mitigation_candidates()
    print("\n=== pareto_and_test ==="); pareto_and_test()
    print("\n=== explanation_runs ==="); explanation_runs()
    print("\n=== explanation_metrics ==="); explanation_metrics()
    print("\n=== plot_shap ==="); plot_shap()
    print("\n=== reproducibility_log ==="); reproducibility_log()
    print("\n=== paper_vs_reproduced ==="); paper_vs_reproduced()
    print("\n=== export_web_results ==="); export_web_results()

if __name__=="__main__":
    main()
