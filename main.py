from utils import *

cfg = read_config()
df = read_data(f"{cfg.data.path}-preprocessed.csv")

plot_wordcloud(df, key='Text', fig_size=(8,6))