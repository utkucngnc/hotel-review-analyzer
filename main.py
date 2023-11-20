from utils import *
from basic import BasicClassifier as bc

cfg = read_config()
df_preprocessed = read_data(cfg.data_preprocessed.path)

clsfr = bc(df_preprocessed, cfg.data_preprocessed.heads)
model = clsfr.load_model(cfg.model.save_path)

print(model.predict(["bad"]))