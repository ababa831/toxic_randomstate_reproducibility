# toxic_randomforest_reproducibility

とある条件下でランダムフォレストの推論再現性を検証
[Kaggle Toxicコンペのデータ](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)を使用

一部の人向け

# Usage

0. Download dataset of Kaggle Toxic competition, and put it on the directory.
1. `$ pip install pytest` (If necessary)
2. `$ pytest -vv -s classification.py`, and several reproducibility tests will run.
