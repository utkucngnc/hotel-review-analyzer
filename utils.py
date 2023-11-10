import pandas as pd
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

def read_config(path: str = "./config.yaml") -> dict:
    config = OmegaConf.load(path)
    
    return config

def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.dropna(axis=0,inplace=True)
    
    return data

def save_data(data: pd.DataFrame, save_path: str) -> None:
    data.to_csv(save_path, index=False)

def plot_distribution(data: pd.DataFrame, keys: list, fig_size: tuple = (8,6)) -> None:
    plt.figure(figsize=fig_size)
    plt.subplots_adjust(hspace=0.2)
    plt.suptitle("Data Distribution", fontsize=18, y=0.95)
    
    ncols = 2
    nrows = len(keys) // ncols + (len(keys) % ncols > 0)
    
    for i, key in enumerate(keys):
        ax = plt.subplot(nrows, ncols, i+1)
        
        data[key].hist(ax=ax, bins=50)
        ax.set_title(key)
    
    plt.show()
    