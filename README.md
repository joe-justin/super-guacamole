METRICS.CSV Format:
Function, Application, Server, timestamp, cpu_current, cpu_min, cpu_max, mem_current,

pip install pandas numpy scikit-learn torch statsmodels tqdm
python model_pipeline_multi.py --input metrics.csv --outdir output --forecast_days 30 --gru_epochs 50


cd backend
export DATA_DIR="../output"   # or set in environment accordingly
uvicorn main:app --reload --port 8000


cd frontend
npm install
npm run dev
# open http://localhost:5173
