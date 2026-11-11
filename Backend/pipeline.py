"""Q-BIRD reproducible Atomiq pipeline. Seeded, auditable, and honest about simulator availability."""
from pathlib import Path
import json, itertools, math, time, platform, sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, precision_score, recall_score
from scipy.optimize import minimize

SEED=42
BUDGET=25
CAPACITY=22
K=6

def make_model(kind, X):
    cats=X.select_dtypes(include=['object','category','bool']).columns.tolist()
    nums=[c for c in X.columns if c not in cats]
    try: enc=OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError: enc=OneHotEncoder(handle_unknown='ignore', sparse=True)
    prep=ColumnTransformer([
      ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
      ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',enc)]),cats)
    ])
    if kind=='logistic_regression': clf=LogisticRegression(max_iter=2500,class_weight='balanced',C=0.35,random_state=SEED)
    elif kind=='random_forest': clf=RandomForestClassifier(n_estimators=300,min_samples_leaf=3,max_features='sqrt',class_weight='balanced_subsample',random_state=SEED,n_jobs=-1)
    else: clf=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,max_features='sqrt',class_weight='balanced',random_state=SEED,n_jobs=-1)
    return Pipeline([('preprocess',prep),('model',clf)])

def exact_poisson_binomial(probs):
    dist=np.array([1.0])
    for p in probs: dist=np.convolve(dist,[1-float(p),float(p)])
    return float(dist[3:].sum()),dist.tolist()

def qaoa_statevector(values,costs,units,budget=25,capacity=22,shots=8192):
    """p=1 QAOA-style diagonal-Hamiltonian statevector experiment on 10 decision qubits.
    This is a custom statevector implementation, not a Qiskit-transpiled Stage-2 circuit.
    Stage-3 uses Qiskit Aer when installed. Exact enumeration remains the correctness oracle.
    """
    n=len(values); basis=np.array(list(itertools.product([0,1],repeat=n)),dtype=np.int8)
    cnt=basis.sum(axis=1); spent=basis@costs; cap=basis@units; utility=basis@values
    # Penalize cardinality and positive inequality violations; scale penalties above the objective range.
    energy=-utility + 80.0*(cnt-K)**2 + 80.0*np.maximum(0,spent-budget)**2 + 80.0*np.maximum(0,cap-capacity)**2
    def state(params):
        gamma,beta=params
        a=np.ones(2**n,dtype=complex)/math.sqrt(2**n)
        a*=np.exp(-1j*gamma*energy)
        c,s=np.cos(beta),-1j*np.sin(beta)
        mixer=np.array([[c,s],[s,c]],complex)
        for q in range(n):
            a=a.reshape([2]*n); a=np.moveaxis(a,q,0)
            a=np.tensordot(mixer,a,axes=(1,0)); a=np.moveaxis(a,0,q).reshape(-1)
        return a
    def objective(params):
        a=state(params); return float(np.sum(np.abs(a)**2*energy))
    best=None
    for init in ([0.22,0.31],[0.55,0.75],[0.85,0.28]):
        res=minimize(objective,np.array(init,dtype=float),method='COBYLA',options={'maxiter':100,'rhobeg':0.25,'tol':1e-4})
        if best is None or res.fun<best.fun: best=res
    amplitudes=state(best.x); dist=np.abs(amplitudes)**2; dist=dist/dist.sum()
    rng=np.random.default_rng(SEED); draws=rng.choice(len(basis),size=shots,p=dist)
    counts={}
    for idx in draws:
        bit=''.join(map(str,basis[idx].tolist())); counts[bit]=counts.get(bit,0)+1
    measured=[]
    for bit,num in counts.items():
        x=np.fromiter(map(int,bit),dtype=int)
        if x.sum()==K and x@costs<=budget and x@units<=capacity:
            measured.append((float(x@values),bit,num,float(x@costs),int(x@units)))
    # Keep the best feasible measured sample. If finite shots missed all feasible states,
    # report the highest-probability feasible state and disclose the fallback in metadata.
    fallback=False
    if measured:
        measured.sort(key=lambda z:(-z[0],z[1])); chosen=measured[0]
    else:
        feasible=np.where((cnt==K)&(spent<=budget)&(cap<=capacity))[0]
        if not len(feasible): raise ValueError('No feasible selection exists under the official constraints.')
        idx=max(feasible,key=lambda i:dist[i]); bit=''.join(map(str,basis[idx].tolist()))
        chosen=(float(utility[idx]),bit,0,float(spent[idx]),int(cap[idx])); fallback=True
    return {'bitstring':chosen[1],'objective':chosen[0],'budget_used':chosen[3],'capacity_used':chosen[4],
            'angles':best.x.tolist(),'measurement_counts':counts,'shots':shots,'fallback_to_most_likely_feasible_state':fallback,
            'method':'custom p=1 diagonal-Hamiltonian statevector QAOA; Stage-2 circuit not transpiled by Qiskit',
            'feasible_measured_samples':len(measured)}

def load_official(source):
    p=Path(source)
    if p.suffix.lower() in ('.xlsx','.xls'):
        train=pd.read_excel(p,sheet_name='Stage1_Train'); test=pd.read_excel(p,sheet_name='Stage1_Test'); ops=pd.read_excel(p,sheet_name='Stage2_Data')
    else:
        # Also accept a pre-flattened CSV for reproducible runs.
        df=pd.read_csv(p)
        if {'split','y','customer_id'}.issubset(df.columns):
            train=df[df.split.astype(str).str.lower().isin(['train','training','0'])].copy(); test=df[df.split.astype(str).str.lower().isin(['test','testing','1'])].copy()
            ops=pd.read_csv(Path(p).parent/'stage2_data.csv')
        else: raise ValueError('Use the official XLSX workbook or a flattened CSV with split plus stage2_data.csv.')
    return train,test,ops

def run_pipeline(source,budget=BUDGET,capacity=CAPACITY,out_dir=None):
    started=time.time(); root=Path(out_dir or Path(__file__).resolve().parents[1]); root.mkdir(parents=True,exist_ok=True)
    if float(budget)!=BUDGET or int(capacity)!=CAPACITY:
        # Challenge constraints are fixed by the supplied statement; don't silently change them.
        raise ValueError('Official challenge constraints are fixed: budget=25 and capacity=22.');
    train,test,ops=load_official(source)
    required_train={'customer_id','y'}; required_ops={'customer_id','contact_cost','expected_deposit_value','capacity_units','customer_segment'}
    if not required_train.issubset(train.columns) or 'customer_id' not in test: raise ValueError('Official workbook is missing required Stage 1 columns.')
    if len(train)!=300 or len(test)!=100: raise ValueError(f'Expected 300 train and 100 test rows; got {len(train)} and {len(test)}.')
    features=[c for c in train.columns if c not in ('customer_id','y') and c in test.columns]
    X=train[features].copy(); y=train.y.astype(str).str.strip().str.lower().map({'yes':1,'no':0})
    if y.isna().any(): raise ValueError('Training target y contains values other than yes/no.')
    # Same seeded folds for all candidate models; all preprocessing lives inside each fold's pipeline.
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED); metrics={}; models={}; cvpred={}
    for name in ('logistic_regression','random_forest','extra_trees'):
        model=make_model(name,X)
        try:
            pred=cross_val_predict(model,X,y,cv=cv,method='predict_proba',n_jobs=1)[:,1]
            metrics[name]={'roc_auc':float(roc_auc_score(y,pred)),'pr_auc':float(average_precision_score(y,pred)),
              'brier_score':float(brier_score_loss(y,pred)),'log_loss':float(log_loss(y,pred,labels=[0,1])),
              'precision_at_10':float(y.iloc[np.argsort(-pred,kind='stable')[:10]].mean()),
              'recall_at_10':float(y.iloc[np.argsort(-pred,kind='stable')[:10]].sum()/max(int(y.sum()),1)),
              'evaluation':'5-fold out-of-fold predictions on training rows'}
            models[name]=model; cvpred[name]=pred
        except Exception as e: metrics[name]={'status':'failed','reason':str(e)}
    valid=[k for k,v in metrics.items() if 'pr_auc' in v]
    if not valid: raise RuntimeError('All Stage 1 model evaluations failed.')
    winner=max(valid,key=lambda k:(metrics[k]['pr_auc'],-metrics[k]['brier_score']))
    final=models[winner].fit(X,y); pred=final.predict_proba(test[features])[:,1]
    # Official deterministic tie-breaking: probability descending, customer ID ascending alphanumeric.
    pred_df=pd.DataFrame({'customer_id':test.customer_id.astype(str),'predicted_probability':np.clip(pred,0,1)})
    pred_df=pred_df.sort_values(['predicted_probability','customer_id'],ascending=[False,True],kind='mergesort').reset_index(drop=True)
    pred_df['rank']=np.arange(1,len(pred_df)+1); pred_df['is_top_10']=pred_df['rank']<=10
    pred_df.to_csv(root/'stage1_predictions.csv',index=False,float_format='%.10f')
    top=pred_df.head(10).merge(ops,on='customer_id',how='left',validate='one_to_one')
    if top[['contact_cost','expected_deposit_value','capacity_units']].isna().any().any(): raise ValueError('Stage2_Data is missing operational data for one or more top-10 customers.')
    top['expected_net_value']=top.predicted_probability*top.expected_deposit_value-top.contact_cost
    values=top.expected_net_value.to_numpy(float); costs=top.contact_cost.to_numpy(float); units=top.capacity_units.to_numpy(float)
    combos=[]
    for ix in itertools.combinations(range(10),K):
        spend=float(costs[list(ix)].sum()); cap=float(units[list(ix)].sum())
        if spend<=budget and cap<=capacity:
            combos.append((float(values[list(ix)].sum()),ix,spend,cap))
    if not combos: raise ValueError('No feasible six-customer subset satisfies both official constraints.')
    combos.sort(key=lambda z:(-z[0],tuple(top.iloc[i].customer_id for i in z[1])))
    optimum,optimal_ix,opt_cost,opt_cap=combos[0]
    qresult=qaoa_statevector(values,costs,units,budget,capacity,shots=8192)
    selected_ix=[i for i,b in enumerate(qresult['bitstring']) if b=='1']
    selected=top.iloc[selected_ix].copy()
    # If QAOA sampling missed or returned a lower-value solution, preserve its honest result and also include exact optimum.
    qobj=float(selected.expected_net_value.sum()); qcost=float(selected.contact_cost.sum()); qcap=float(selected.capacity_units.sum())
    selected.to_csv(root/'stage2_selection.csv',index=False,float_format='%.10f')
    gap=float((optimum-qobj)/max(abs(optimum),1e-12))
    stage2={**qresult,'top_10_candidate_ids':top.customer_id.tolist(),'selected_customer_ids':selected.customer_id.tolist(),
      'selected_count':len(selected),'budget_limit':float(budget),'budget_used':qcost,'capacity_limit':int(capacity),'capacity_used':qcap,
      'objective_value':qobj,'objective_definition':'maximize sum(predicted_probability * expected_deposit_value - contact_cost)',
      'classical_bruteforce_optimum':float(optimum),'classical_optimal_customer_ids':[top.iloc[i].customer_id for i in optimal_ix],
      'classical_budget_used':opt_cost,'classical_capacity_used':opt_cap,'feasible_combinations_evaluated':len(combos),
      'optimality_gap':gap,'constraints_satisfied':bool(len(selected)==6 and qcost<=budget and qcap<=capacity),
      'selected_rows':selected[['customer_id','predicted_probability','contact_cost','expected_deposit_value','capacity_units','expected_net_value']].to_dict(orient='records')}
    (root/'stage2_selection.json').write_text(json.dumps(stage2,indent=2))
    # Stage 3: use Qiskit Aer if installed. The fallback is a seeded Bernoulli shot simulator and is labelled honestly.
    probs=selected.predicted_probability.astype(float).tolist(); shots=8192; backend='numpy seeded Bernoulli fallback (Qiskit Aer unavailable)'; circuit_meta={}; noisy_counts={}; noisy_prob=None
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        qc=QuantumCircuit(6,6)
        for i,pv in enumerate(probs): qc.ry(2*math.asin(math.sqrt(float(pv))),i)
        qc.measure(range(6),range(6))
        sim=AerSimulator(); tqc=transpile(qc,sim,seed_transpiler=SEED)
        counts=sim.run(tqc,shots=shots,seed_simulator=SEED).result().get_counts()
        success=sum(v for bits,v in counts.items() if bits.replace(' ','').count('1')>=3); qprob=success/shots
        noise=NoiseModel(); noise.add_all_qubit_quantum_error(depolarizing_error(0.001,1),['ry'])
        noisy_sim=AerSimulator(noise_model=noise); ntqc=transpile(qc,noisy_sim,seed_transpiler=SEED)
        noisy_counts=noisy_sim.run(ntqc,shots=shots,seed_simulator=SEED+1).result().get_counts()
        noisy_prob=sum(v for bits,v in noisy_counts.items() if bits.replace(' ','').count('1')>=3)/shots
        backend='Qiskit AerSimulator'; circuit_meta={'transpiled_depth':tqc.depth(),'gate_counts':{str(k):int(v) for k,v in tqc.count_ops().items()},'logical_qubits':6,'physical_qubits':tqc.num_qubits}
    except ImportError:
        rng=np.random.default_rng(SEED); bit_draws=(rng.random((shots,6))<np.array(probs)).astype(int)
        qprob=float(np.mean(bit_draws.sum(axis=1)>=3)); counts={''.join(map(str,row)):int(n) for row,n in zip(*np.unique(bit_draws,axis=0,return_counts=True))}
        # Explicit approximate bit-flip noise fallback, not a gate-level depolarizing simulation.
        rng2=np.random.default_rng(SEED+1); noisy_bits=bit_draws.copy(); noisy_bits ^= (rng2.random((shots,6))<0.001).astype(int)
        noisy_prob=float(np.mean(noisy_bits.sum(axis=1)>=3)); noisy_counts={''.join(map(str,row)):int(n) for row,n in zip(*np.unique(noisy_bits,axis=0,return_counts=True))}
        backend='NumPy seeded shot simulator with approximate 0.1% measurement bit-flip noise (Qiskit Aer unavailable)'
        circuit_meta={'logical_qubits':6,'physical_qubits':6,'transpiled_depth':None,'gate_counts':{'ry':6,'measure':6},'note':'Analytical circuit specification; no Qiskit transpilation occurred.'}
    exact,dist=exact_poisson_binomial(probs)
    rng=np.random.default_rng(SEED+2); mc=float(np.mean((rng.random((500000,6))<np.array(probs)).sum(axis=1)>=3))
    stage3={'selected_customer_ids':selected.customer_id.tolist(),'customer_to_qubit_mapping':{str(row.customer_id):i for i,row in enumerate(selected.itertuples())},
      'input_probabilities':probs,'circuit':'6-qubit independent Bernoulli state preparation using RY(2 asin(sqrt(p_i))) followed by measurement',
      'measurement_counts':counts,'shots':shots,'quantum_estimated_probability_ge_3':qprob,'exact_analytical_probability_ge_3':exact,
      'monte_carlo_probability_ge_3':mc,'noisy_simulation_probability_ge_3':noisy_prob,'noise_model':({'type':'single-qubit depolarizing error','probability':0.001,'affected_gate':['ry']} if backend=='Qiskit AerSimulator' else {'type':'approximate measurement bit-flip','probability':0.001,'note':'Fallback approximation only; not a gate-level depolarizing noise model'}),
      'noisy_measurement_counts':noisy_counts,'noisy_absolute_error':abs(noisy_prob-exact),'quantum_absolute_error':abs(qprob-exact),
      'monte_carlo_absolute_error':abs(mc-exact),'execution_backend':backend,'seed':SEED,'seed_shots':SEED,'seed_noise':SEED+1,'seed_monte_carlo':SEED+2,'monte_carlo_iterations':500000,'transpiler_seed':SEED if backend=='Qiskit AerSimulator' else None,**circuit_meta}
    stage3['simulation_note']='The independent Bernoulli RY circuit encodes input probabilities; it is a probability-estimation circuit, not evidence of quantum advantage.'
    (root/'stage3_simulation.json').write_text(json.dumps(stage3,indent=2))
    pd.DataFrame([{'selected_ids':'|'.join(stage3['selected_customer_ids']),'customer_to_qubit_mapping':json.dumps(stage3['customer_to_qubit_mapping']),
      'input_probabilities':json.dumps(probs),'quantum_estimated_probability_ge_3':qprob,'exact_analytical_probability_ge_3':exact,
      'monte_carlo_probability_ge_3':mc,'noisy_simulation_probability_ge_3':noisy_prob,'noisy_absolute_error':abs(noisy_prob-exact),
      'quantum_absolute_error':abs(qprob-exact),'monte_carlo_absolute_error':abs(mc-exact),'shots':shots,'execution_backend':backend,
      'transpiled_depth':circuit_meta.get('transpiled_depth'),'gate_counts':json.dumps(circuit_meta.get('gate_counts',{}))}]).to_csv(root/'stage3_simulation.csv',index=False)
    meta={'seed':SEED,'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'sklearn':__import__('sklearn').__version__,
      'training_rows':len(train),'test_rows':len(test),'features':features,'positive_training_labels':int(y.sum()),'selected_model':winner,
      'validation_metrics':metrics,'stage2_method':qresult['method'],'stage2_seed':SEED,'stage2_shots':qresult['shots'],'stage3_backend':backend,'stage3_shots':shots,'stage3_seed':SEED,'stage3_monte_carlo_iterations':500000,'runtime_seconds':time.time()-started}
    (root/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    return {'stage1':{'customers':len(pred_df),'top10':10,'selected_model':winner,'validation_metrics':metrics},
      'stage2':{'selected_count':len(selected),'selected_customer_ids':selected.customer_id.tolist(),'objective_value':qobj,'classical_optimum':optimum,'optimality_gap':gap,'budget_used':qcost,'capacity_used':qcap},
      'stage3':{'p_at_least_3':qprob,'exact_probability':exact,'monte_carlo_probability':mc,'backend':backend},'metadata':meta}
