from mksc.core import reader
import os

def save_result(data, filename, remote=False):
    cfg = reader.config()
    if remote:
        data.to_sql(cfg.get('DATABASE', 'SCHEMA_NAME'), cfg.get('DATABASE', 'SAVE_ENGINE_URL'), if_exists='replace')
    else:
        data.to_csv(os.path.join(cfg.get('PATH', 'SAVE_DIR'), filename), index=False)