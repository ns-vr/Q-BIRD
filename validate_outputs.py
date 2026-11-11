"""Fast consistency checks for Q-BIRD generated challenge artifacts."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent
p1=pd.read_csv(ROOT/'stage1_predictions.csv')
p2=pd.read_csv(ROOT/'stage2_selection.csv')
s2=json.loads((ROOT/'stage2_selection.json').read_text())
s3=json.loads((ROOT/'stage3_simulation.json').read_text())
assert len(p1)==100 and p1.customer_id.nunique()==100
assert list(p1.columns)==['customer_id','predicted_probability','rank','is_top_10']
assert p1.predicted_probability.between(0,1).all()
assert int(p1.is_top_10.sum())==10
assert p1["rank"].tolist()==list(range(1,101))
assert len(p2)==6 and p2.customer_id.nunique()==6
assert set(p2.customer_id).issubset(set(p1.loc[p1.is_top_10,'customer_id']))
assert p2.contact_cost.sum()<=25 and p2.capacity_units.sum()<=22
assert abs(p2.expected_net_value.sum()-s2['objective_value'])<1e-6
assert s2['constraints_satisfied'] is True
assert s2['selected_customer_ids']==s3['selected_customer_ids']
assert len(s3['input_probabilities'])==6
assert all(0<=float(p)<=1 for p in s3['input_probabilities'])
assert 0<=s3['quantum_estimated_probability_ge_3']<=1
assert 0<=s3['exact_analytical_probability_ge_3']<=1
print('PASS: Stage 1 dimensions, ranking and probabilities')
print('PASS: Stage 2 cardinality, traceability, budget, capacity and objective')
print('PASS: Stage 3 six-probability input and probability ranges')
print('Note: inspect stage3_simulation.json execution_backend before describing quantum execution.')
