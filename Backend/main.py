from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile, shutil
try:
    from .pipeline import run_pipeline
except ImportError:
    from pipeline import run_pipeline

ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title='Q-BIRD API',version='2.0.0',description='Reproducible hybrid quantum-classical banking campaign prototype')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173','null'],allow_credentials=False,allow_methods=['*'],allow_headers=['*'])
@app.get('/health')
def health(): return {'status':'ok','service':'q-bird-api','official_dataset_configured':(ROOT/'data'/'bank_campaign_participant_release.xlsx').exists()}
@app.post('/pipeline/run')
async def pipeline_run(file: UploadFile|None=File(default=None),budget:float=Form(25),capacity:int=Form(22)):
    source=None; temporary=False
    try:
        if file:
            suffix=Path(file.filename or '').suffix.lower()
            if suffix not in ('.xlsx','.xls','.csv'): raise HTTPException(400,'Upload the official .xlsx workbook or a supported .csv.')
            with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
                shutil.copyfileobj(file.file,tmp); source=Path(tmp.name); temporary=True
        else: source=ROOT/'data'/'bank_campaign_participant_release.xlsx'
        if not source.exists(): raise HTTPException(400,'Official workbook missing from data/.')
        return run_pipeline(source,budget=budget,capacity=capacity,out_dir=ROOT)
    except HTTPException: raise
    except Exception as e: raise HTTPException(500,f'Pipeline error: {type(e).__name__}: {e}')
    finally:
        if temporary and source: source.unlink(missing_ok=True)
@app.get('/results/{name}')
def get_result(name:str):
    allowed={'stage1_predictions.csv','stage2_selection.csv','stage2_selection.json','stage3_simulation.csv','stage3_simulation.json','run_metadata.json'}
    if name not in allowed: raise HTTPException(404,'Unknown result file')
    p=ROOT/name
    if not p.exists(): raise HTTPException(404,'No result yet. Run the pipeline first.')
    return FileResponse(p,filename=name)
