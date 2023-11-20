from utils import *

cfg = read_config()
df_preprocessed = read_data(cfg.data_preprocessed.path)

plot_wordcloud(df_preprocessed)