from mksc.core import reader

def save_result(data, mode="local"):
    cfg = reader.config()
    if mode == "local":
        data.to_csv(os.path.join(cfg.get('PATH', 'save_dir'), 'apply_result.csv', index=False)
    else:
        data.to_sql(cfg.get('DATABASE', 'schema_name'), cfg.get('DATABASE', 'engine_url'), if_exists='replace')