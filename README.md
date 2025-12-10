twitter-sentiment/
├─ data/
│  ├─ raw/                   # original dataset (sentiment140 CSV)
│  └─ processed/             # cleaned & tokenized data
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  ├─ 02_preprocessing.ipynb
│  └─ 03_modeling_baselines.ipynb
├─ src/
│  ├─ data_preprocessing.py
│  ├─ features.py
│  ├─ train_baseline.py
│  ├─ train_lstm.py
│  ├─ train_transformer.py
│  └─ evaluate.py
├─ models/                   # saved model checkpoints
├─ app/
│  ├─ streamlit_app.py       # simple demo UI
│  └─ flask_api.py
├─ reports/
│  └─ final_report.pdf       # final project report (use template)
├─ requirements.txt
└─ README.md
