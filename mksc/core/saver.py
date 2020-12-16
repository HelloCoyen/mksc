from mksc.core import reader
import os

def save_result(data, filename=None, remote=False):
    cfg = reader.config()
    if remote:
        data.to_sql(cfg.get('DATABASE', 'SCHEMA_NAME'), cfg.get('DATABASE', 'SAVE_ENGINE_URL'), if_exists='replace')
    data.to_csv(os.path.join(os.getcwd(), 'result', filename), index=False)
