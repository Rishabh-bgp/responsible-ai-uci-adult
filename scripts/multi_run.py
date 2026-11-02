"""Repeated-runs experiment: reruns split + training + validation thresholds + test metrics + fairness gaps
+ mitigation choice over several seeds. Writes per-run and summary CSVs and figures to results/multi_run/.

Run:  python scripts/multi_run.py [--seeds 10] [--boot 500]

Design choices (documented, to keep CPU time reasonable):
* Hyperparameters are fixed to the values chosen by 3-fold CV in the single seed-42 run
  (results/reproducibility_log.json) instead of being re-tuned for every seed.
* Each seed changes the stratified 60/20/20 split and every model's random state.
* Bootstrap CIs use --boot resamples per run (default 500, vs 5,000 in the single run).
* The explanation audit (SHAP/LIME/permutation) is NOT repeated: it is the slowest stage, so it stays single-run.
"""
import os, json, time, argparse, urllib.request, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "multi_run")
DATA = os.path.join(ROOT, "data")
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
COLS = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship',
        'race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
NUM = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
CAT = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
MODELS = ['Logistic Regression','Random Forest','XGBoost','Neural Network (MLP)']

def load_data():
    os.makedirs(DATA, exist_ok=True)
    for f in ['adult.data','adult.test']:
        p = os.path.join(DATA, f)
        if not os.path.exists(p): urllib.request.urlretrieve(BASE + f, p)
    tr = pd.read_csv(os.path.join(DATA,'adult.data'), names=COLS, skipinitialspace=True, na_values='?')
    te = pd.read_csv(os.path.join(DATA,'adult.test'), names=COLS, skipinitialspace=True, na_values='?', skiprows=1)
    df = pd.concat([tr, te], ignore_index=True); df['y'] = (df.income.str.rstrip('.') == '>50K').astype(int)
    return df

def prep():
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUM),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CAT)])

def make_models(seed, bp):
    g = lambda m, k: bp[m][k]
    return {
     'Logistic Regression': LogisticRegression(max_iter=2000, C=float(g('Logistic Regression','clf__C')), random_state=seed),
     'Random Forest': RandomForestClassifier(n_estimators=int(g('Random Forest','clf__n_estimators')), max_depth=int(g('Random Forest','clf__max_depth')),
                        min_samples_leaf=int(g('Random Forest','clf__min_samples_leaf')), random_state=seed, n_jobs=-1),
     'XGBoost': XGBClassifier(n_estimators=int(g('XGBoost','clf__n_estimators')), max_depth=int(g('XGBoost','clf__max_depth')),
                        learning_rate=float(g('XGBoost','clf__learning_rate')), random_state=seed, n_jobs=-1, eval_metric='logloss', tree_method='hist'),
     'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=eval(g('Neural Network (MLP)','clf__hidden_layer_sizes')),
                        alpha=float(g('Neural Network (MLP)','clf__alpha')), max_iter=300, early_stopping=True, random_state=seed)}

def best_thr(p, y):
    ts = np.linspace(0.05, 0.95, 181); return float(ts[np.argmax([f1_score(y, p >= t) for t in ts])])

def gaps(y, yh, g):
    m, f = g == 'Male', g == 'Female'
    return [abs(yh[m].mean()-yh[f].mean()), abs(yh[m&(y==1)].mean()-yh[f&(y==1)].mean()), abs(yh[m&(y==0)].mean()-yh[f&(y==0)].mean())]

class F64(BaseEstimator, ClassifierMixin):
    """float64 wrapper: avoids a fairlearn/pandas float32 setitem error."""
    def __init__(self, m): self.m = m
    def fit(self, X, y): return self
    def predict_proba(self, X): return self.m.predict_proba(X).astype('float64')
    def predict(self, X): return self.m.predict(X)
    def __sklearn_is_fitted__(self): return True

def run_seed(df, seed, bp, n_boot):
    t0 = time.time(); rng = np.random.default_rng(seed)
    X = df[NUM+CAT]; y = df.y.values; strat = df.y.astype(str) + df.sex
    Xtr, Xtmp, ytr, ytmp, _, stmp = train_test_split(X, y, strat, test_size=0.4, stratify=strat, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=stmp, random_state=seed)
    s_tr, s_va, s_te = Xtr.sex.values, Xva.sex.values, Xte.sex.values
    strata = [np.where((s_te==s)&(yte==c))[0] for s in ['Male','Female'] for c in [0,1]]
    boot = [np.concatenate([rng.choice(ix, len(ix)) for ix in strata]) for _ in range(n_boot)]
    rows, fitted, Pva, thr = [], {}, {}, {}
    for name, clf in make_models(seed, bp).items():
        m = Pipeline([('prep', prep()), ('clf', clf)]).fit(Xtr, ytr); fitted[name] = m
        Pva[name] = m.predict_proba(Xva)[:,1]; thr[name] = best_thr(Pva[name], yva)
        pt = m.predict_proba(Xte)[:,1]; yh = (pt >= thr[name]).astype(int); g = gaps(yte, yh, s_te)
        bs = np.array([gaps(yte[ix], yh[ix], s_te[ix]) for ix in boot]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rows.append(dict(seed=seed, model=name, threshold=thr[name], accuracy=accuracy_score(yte, yh), precision=precision_score(yte, yh),
             recall=recall_score(yte, yh), f1=f1_score(yte, yh), roc_auc=roc_auc_score(yte, pt),
             DPD=g[0], DPD_lo=lo[0], DPD_hi=hi[0], EOD=g[1], EOD_lo=lo[1], EOD_hi=hi[1], FPRD=g[2], FPRD_lo=lo[2], FPRD_hi=hi[2]))
    dpd = lambda yh, g: abs(yh[g=='Male'].mean()-yh[g=='Female'].mean())
    cands = {f'{n} (baseline)': ((Pva[n]>=thr[n]).astype(int), lambda n=n: (fitted[n].predict_proba(Xte)[:,1]>=thr[n]).astype(int)) for n in MODELS}
    pg = pd.crosstab(s_tr, ytr, normalize=True); w = np.array([pg.loc[s].sum()*pg[c].sum()/pg.loc[s,c] for s, c in zip(s_tr, ytr)])
    rw = clone(fitted['XGBoost']).fit(Xtr, ytr, clf__sample_weight=w); prw = rw.predict_proba(Xva)[:,1]; trw = best_thr(prw, yva)
    cands['XGBoost + reweighing'] = ((prw>=trw).astype(int), lambda: (rw.predict_proba(Xte)[:,1]>=trw).astype(int))
    pre = prep().fit(Xtr); Ztr, Zva, Zte = pre.transform(Xtr), pre.transform(Xva), pre.transform(Xte)
    for eps in [0.01, 0.02, 0.05, 0.1]:
        np.random.seed(seed)
        eg = ExponentiatedGradient(LogisticRegression(max_iter=2000, C=float(bp['Logistic Regression']['clf__C'])), DemographicParity(difference_bound=eps), eps=eps)
        eg.fit(Ztr, ytr, sensitive_features=s_tr)
        cands[f'ExpGrad-LR eps={eps}'] = (eg.predict(Zva, random_state=seed), lambda eg=eg: eg.predict(Zte, random_state=seed))
    x64 = F64(fitted['XGBoost']); x64.classes_ = np.array([0,1])
    for cons in ['demographic_parity','equalized_odds','true_positive_rate_parity']:
        to = ThresholdOptimizer(estimator=x64, constraints=cons, objective='accuracy_score', prefit=True, predict_method='predict_proba').fit(Xva, yva, sensitive_features=s_va)
        cands[f'XGBoost + ThresholdOpt ({cons})'] = (to.predict(Xva, sensitive_features=s_va, random_state=seed), lambda to=to: to.predict(Xte, sensitive_features=s_te, random_state=seed))
    pv = pd.DataFrame([dict(candidate=k, val_acc=accuracy_score(yva, v[0]), val_DPD=dpd(v[0], s_va)) for k, v in cands.items()]).set_index('candidate')
    ok = pv[pv.val_DPD <= 0.05]; chosen = ok.val_acc.idxmax() if len(ok) else pv.val_DPD.idxmin()
    yc = cands[chosen][1](); g = gaps(yte, yc, s_te)
    mit = dict(seed=seed, chosen=chosen, val_acc=float(pv.loc[chosen,'val_acc']), val_DPD=float(pv.loc[chosen,'val_DPD']),
               test_accuracy=accuracy_score(yte, yc), test_f1=f1_score(yte, yc), test_DPD=g[0], test_EOD=g[1], test_FPRD=g[2])
    print(f'seed {seed}: done in {time.time()-t0:.0f}s, chosen = {chosen}', flush=True)
    return rows, mit

def summarise(per):
    met = ['accuracy','precision','recall','f1','roc_auc','DPD','EOD','FPRD']
    s = per.groupby('model')[met].agg(['mean','std','min','max']).reindex(MODELS)
    s.columns = [f'{a}_{b}' for a, b in s.columns]; return s

def rank_stability(per):
    """How often each model is best (highest accuracy/F1/AUC, lowest gaps) across seeds."""
    out = []
    for m in ['accuracy','f1','roc_auc','DPD','EOD','FPRD']:
        asc = m in ('DPD','EOD','FPRD')
        best = per.sort_values(m, ascending=asc).groupby('seed').head(1).model.value_counts()
        out.append(dict(metric=m, best_is='lowest' if asc else 'highest', **{k: int(best.get(k, 0)) for k in MODELS}))
    return pd.DataFrame(out)

def figures(per):
    short = {'Logistic Regression':'LogReg','Random Forest':'RF','XGBoost':'XGBoost','Neural Network (MLP)':'MLP'}
    for group, mets, fn in [('performance', ['accuracy','f1','roc_auc'], 'box_performance.png'), ('fairness gaps by sex', ['DPD','EOD','FPRD'], 'box_fairness.png')]:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4)); jit = np.random.default_rng(0)
        for ax, m in zip(axs, mets):
            ax.boxplot([per[per.model==n][m] for n in MODELS], tick_labels=[short[n] for n in MODELS])
            for i, n in enumerate(MODELS):
                v = per[per.model==n][m]; ax.scatter(np.full(len(v), i+1)+jit.uniform(-.08,.08,len(v)), v, s=12, alpha=.6)
            ax.set_title(m.upper().replace('_','-')); ax.grid(alpha=.3)
        fig.suptitle(f'Test-set {group} across {per.seed.nunique()} seeds'); plt.tight_layout(); plt.savefig(os.path.join(OUT, fn), dpi=130); plt.close()

def main(seeds=10, n_boot=500):
    os.makedirs(OUT, exist_ok=True); T = time.time()
    bp = json.load(open(os.path.join(ROOT, 'results', 'reproducibility_log.json')))['best_params']
    df = load_data(); allrows, mits = [], []
    for s in range(seeds):
        r, m = run_seed(df, s, bp, n_boot); allrows += r; mits.append(m)
    per = pd.DataFrame(allrows); mit = pd.DataFrame(mits); summ = summarise(per); rk = rank_stability(per)
    per.to_csv(os.path.join(OUT, 'per_run.csv'), index=False); mit.to_csv(os.path.join(OUT, 'mitigation_per_run.csv'), index=False)
    summ.to_csv(os.path.join(OUT, 'summary.csv')); rk.to_csv(os.path.join(OUT, 'rank_stability.csv'), index=False)
    msum = mit.drop(columns=['seed','chosen']).agg(['mean','std','min','max']).T; msum.to_csv(os.path.join(OUT, 'mitigation_summary.csv'))
    figures(per)
    json.dump(dict(seeds=list(range(seeds)), n_boot_per_run=n_boot, hyperparameters='fixed from the single seed-42 CV run',
                   explanations='not repeated (single-run only)', runtime_sec=round(time.time()-T, 1)), open(os.path.join(OUT, 'config.json'), 'w'), indent=2)
    print(f'total {time.time()-T:.0f}s'); return per, mit, summ, rk, msum

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--seeds', type=int, default=10); a.add_argument('--boot', type=int, default=500)
    o = a.parse_args(); main(o.seeds, o.boot)
