from utils import *

cfg = read_config()
df = read_data(f"{cfg.data.path}-preprocessed.csv")

plot_distribution(df, ["Rating", "Word Count"])