from utils import *
from advanced import AdvancedClassifier

def main():
    cfg = read_config("config.yaml")
    classifier = AdvancedClassifier(cfg)
    classifier.train()
    # classifier.inference()

if __name__ == "__main__":
    main()