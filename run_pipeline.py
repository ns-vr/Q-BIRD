from pathlib import Path
import argparse, json
from backend.pipeline import run_pipeline
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description='Run the Q-BIRD Atomiq three-stage pipeline')
p.add_argument('--workbook',default=str(ROOT/'data'/'bank_campaign_participant_release.xlsx'))
p.add_argument('--budget',type=float,default=25)
p.add_argument('--capacity',type=int,default=22)
a=p.parse_args()
print(json.dumps(run_pipeline(a.workbook,a.budget,a.capacity,ROOT),indent=2))
