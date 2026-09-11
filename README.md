Hiver SDE Intern Take-Home — AI Support Agent (AppleSupport)
Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
Download twcs.csv (Kaggle: thoughtvector/customer-support-on-twitter) into data/raw/.
Step 1: Ingest + reconstruct threads
python src/ingest.py --csv data/raw/twcs.csv --brand AppleSupport
-> data/processed/brand_stats.csv, data/processed/threads.jsonl
(Remaining steps documented as each script is added: taxonomy.py, golden set,
baselines.py, metrics.py, judge.py, retrieve.py, classify.py, generate.py,
escalate.py, pipeline.py.)