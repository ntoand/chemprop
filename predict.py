"""Loads a trained chemprop model checkpoint and makes predictions on a dataset."""

from chemprop.train import chemprop_predict

if __name__ == '__main__':
    preds = chemprop_predict()
    print(preds)
