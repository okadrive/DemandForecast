from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 読み込み
genres = pd.read_csv("genres.csv")
stores = pd.read_csv("stores.csv")
goods = pd.read_csv("goods.csv")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 前処理
# goods と genre の対応付け
dict_goods_to_genre = {}
for index, row in goods.iterrows():
    dict_goods_to_genre[row['goods_id']] = row['goods_genre_id']


def f(x): return dict_goods_to_genre[x]


# store_id と goods_id を合わせた"id"の作成
train["id"] = "store" + train["store_id"].astype("str")
train["id"] = train["id"] + "goods"
train["id"] = train["id"] + train["goods_id"].astype("str")

train["goods_genre_id"] = train['goods_id'].apply(f)

# 年月日の変換
#train['year'] = pd.to_datetime(train['yy_mm_dd'], format='%y-%m-%d').dt.year + 1988 - 2000
#train['month'] = pd.to_datetime(train['yy_mm_dd'], format='%y-%m-%d').dt.month
#train['day'] = pd.to_datetime(train['yy_mm_dd'], format='%y-%m-%d').dt.dayofweek

test["id"] = "store" + test["store_id"].astype("str")
test["id"] = test["id"] + "goods"
test["id"] = test["id"] + test["goods_id"].astype("str")

test["goods_genre_id"] = test['goods_id'].apply(f)

# idごとの月々の売上の算出
id_month_sale = train.groupby(["num_month", "id", "goods_genre_id"]).sum()[
    ["units_sold_day"]].reset_index()
id_month_sale.columns = ['num_month', 'id',
                         'goods_genre_id', 'units_sold_month']

# 予測したい月は"0"なのでnum_monthを0として追加
test["num_month"] = 0

# index, goods_id, store_idは不要なので削除
del test["index"], test["goods_id"], test["store_id"],

# データの整形
data = pd.concat([id_month_sale, test], axis=0).reset_index().iloc[:, 1:]

# 過去12ヶ月の売上をもとに来月の売上を予測するようなshift特徴量の作成
for diff in range(12):
    shift_day = 1 + diff
    print(shift_day)
    data[f"id_shift_t{shift_day}"] = data.groupby(
        ["id"])["units_sold_month"].transform(lambda x: x.shift(shift_day))

print(f"id_shift_t{shift_day}")

# trainとtestに分ける
train_df = data[data["num_month"] != 0]
test_df = data[data["num_month"] == 0]

# NaNを削除
train_df = train_df.dropna()

y = train_df["units_sold_month"]
del train_df["units_sold_month"], test_df["units_sold_month"]

train_df["id"] = train_df["id"].astype("category")
test_df["id"] = test_df["id"].astype("category")

light_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'seed': 42,
    'max_depth': 5,
    'num_leaves': 2**5,
    'learning_rate': 0.1,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "verbose": -1
}


# バリデーション期間を-1とする
# va_period_list = [-1, -2, -3]
va_period_list = [-1]
for va_period in va_period_list:
    tr_idx = train_df["num_month"] < va_period
    val_idx = train_df["num_month"] == va_period
    train_x, test_x = train_df[tr_idx], train_df[val_idx]
    train_y, test_y = y[tr_idx], y[val_idx]

    # Datasetに入れて学習させる
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(test_x, test_y, reference=lgb_train)

    # Training
    model = lgb.train(light_params, lgb_train, num_boost_round=3000, early_stopping_rounds=50,
                      valid_sets=[lgb_train, lgb_valid], verbose_eval=50)

    test_pred = model.predict(test_df)

    oof = model.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, oof))
    print(f"RMSE : {rmse}")

print(rmse)

lgb.plot_importance(model, importance_type="gain",
                    max_num_features=40, figsize=(12, 12))  # max_num_features=20,

sub = pd.read_csv("submission.csv").iloc[:, 1:]
sub["units_sold_month"] = test_pred.round(3)
sub.to_csv("baseline.csv", index=False)
