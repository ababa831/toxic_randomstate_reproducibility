# Ref: https://www.kaggle.com/sudhirnl7/logistic-regression-tfidf
import random
import gc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection
import sklearn.ensemble

PATH_CSV_DATA = './train_toxic.csv'


def tfidf_vectorize(texts, sparse=False):
    vect_word = TfidfVectorizer(
        max_features=20000,
        lowercase=True,
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 3),
        dtype=np.float32)
    tfidf_vec = vect_word.fit_transform(texts)
    if sparse is False:
        tfidf_vec = tfidf_vec.toarray()
    return tfidf_vec


def concat_with_explanatory(tfidf_vec, target_array):
    concated = np.concatenate([tfidf_vec, target_array], axis=1)
    expected_shape = (tfidf_vec.shape[0],
                      tfidf_vec.shape[1] + target_array.shape[1])
    errmsg = 'concatが正しくできていない'
    assert concated.shape == expected_shape, errmsg
    return concated


def change_value_for_some_target_samples(target_array, n_change=200):
    target_array_w_noise = target_array.copy()
    len_target = target_array_w_noise.shape[0]
    # クソコードだけど知るか
    for i in range(target_array_w_noise.shape[1]):
        for _ in range(n_change):
            row = random.randint(0, len_target - 1)
            target_array_w_noise[row, i] = random.randint(0, 1)
    errmsg = '値の置換ができていない'
    assert not (target_array == target_array_w_noise).all(), errmsg
    del target_array
    gc.collect()
    return target_array_w_noise


def evaluate_randomstate_reproducibility(X, y, test_size=0.2):
    train_x, test_x, train_y, test_y = \
        sklearn.model_selection.train_test_split(X,
                                                 y,
                                                 test_size=test_size)

    rf = sklearn.ensemble.RandomForestClassifier()
    rf.fit(train_x, train_y)

    n_trial = 499
    y_pred_1st = rf.predict(test_x)
    errmsg = '同一モデル，入力データで推論したら値が違っちゃったぜ'
    for i in range(n_trial):
        assert (y_pred_1st == rf.predict(test_x)).all(), errmsg


def test_repuroducibility_1():
    """説明変数を目的変数に入れて学習させたとき，推論値は安定するか？"""
    df = pd.read_csv(PATH_CSV_DATA)
    df = df.sample(n=2000)  # 全部使うとメモリが死ぬので
    tfidf_vec = tfidf_vectorize(df['comment_text'])
    target_cols = ['toxic', 'severe_toxic', 'obscene',
                   'threat', 'insult', 'identity_hate']
    target_array = df[target_cols].values
    concated_features = concat_with_explanatory(tfidf_vec, target_array)
    evaluate_randomstate_reproducibility(concated_features, target_array)


def test_repuroducibility_2():
    """ノイズあり説明変数を目的変数に入れる学習させたとき，推論値は安定するか？"""
    df = pd.read_csv(PATH_CSV_DATA)
    df = df.sample(n=2000)  # 全部使うとメモリが死ぬので
    tfidf_vec = tfidf_vectorize(df['comment_text'])
    target_cols = ['toxic', 'severe_toxic', 'obscene',
                   'threat', 'insult', 'identity_hate']
    target_array = df[target_cols].values
    target_array = change_value_for_some_target_samples(target_array)
    concated_features = concat_with_explanatory(tfidf_vec, target_array)
    evaluate_randomstate_reproducibility(concated_features, target_array)
