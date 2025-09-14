import os, glob, re, random, warnings
import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =========================
# 설정
# =========================
LOOKBACK, PREDICT = 28, 7
BATCH_SIZE, EPOCHS = 16, 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 데이터 로드
# =========================
train = pd.read_csv("./train/train.csv")
test_files = sorted(glob.glob("./test/TEST_*.csv"))
sample_submission = pd.read_csv("./sample_submission.csv")

# =========================
# Feature Engineering
# =========================
def add_features(df):
    df["영업일자"] = pd.to_datetime(df["영업일자"])
    df["요일"] = df["영업일자"].dt.weekday
    df["월"] = df["영업일자"].dt.month
    df["주말여부"] = df["요일"].isin([5,6]).astype(int)
    # 이동평균, 표준편차, min/max, 변화율
    df["이동평균"] = df.groupby("영업장명_메뉴명")["매출수량"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["이동표준편차"] = df.groupby("영업장명_메뉴명")["매출수량"].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    df["최소값"] = df.groupby("영업장명_메뉴명")["매출수량"].transform(lambda x: x.rolling(7, min_periods=1).min())
    df["최대값"] = df.groupby("영업장명_메뉴명")["매출수량"].transform(lambda x: x.rolling(7, min_periods=1).max())
    df["증감률"] = df["매출수량"].pct_change().fillna(0)
    return df

train = add_features(train)

# train + test 전체 메뉴를 합쳐서 LabelEncoder 학습
all_menus = pd.concat([train["영업장명_메뉴명"]] + [pd.read_csv(f)["영업장명_메뉴명"] for f in test_files])
le = LabelEncoder()
le.fit(all_menus)

# train에 적용
train["메뉴ID"] = le.transform(train["영업장명_메뉴명"])

# =========================
# Supervised Dataset 생성
# =========================
def make_supervised(df):
    data = []
    for menu_id, group in df.groupby("메뉴ID"):
        group = group.sort_values("영업일자")
        vals = group["매출수량"].values
        extra_feats = group[["요일","월","주말여부","이동평균","이동표준편차","최소값","최대값","증감률"]].values
        for i in range(len(group) - LOOKBACK - PREDICT + 1):
            Xlags = vals[i:i+LOOKBACK]
            feats = extra_feats[i+LOOKBACK-1]
            y = vals[i+LOOKBACK:i+LOOKBACK+PREDICT]
            data.append({
                "메뉴ID": menu_id,
                **{f"lag_{j}": Xlags[j] for j in range(LOOKBACK)},
                "요일": feats[0], "월": feats[1], "주말여부": feats[2],
                "이동평균": feats[3], "이동표준편차": feats[4],
                "최소값": feats[5], "최대값": feats[6], "증감률": feats[7],
                **{f"y_{k+1}": y[k] for k in range(PREDICT)}
            })
    return pd.DataFrame(data)

train_supervised = make_supervised(train)

# =========================
# LightGBM Optuna Tuning
# =========================
def objective_lgbm(trial):
    step = 1
    X = train_supervised.drop([f"y_{k+1}" for k in range(PREDICT)], axis=1)
    y = train_supervised[f"y_{step}"]
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50)
    }
    model = LGBMRegressor(**params)
    model.fit(X, y)
    preds = model.predict(X)
    return np.mean(np.abs((y - preds) / (np.abs(y)+np.abs(preds)+1e-8)))

print(" LGBM Optuna 튜닝중...")
study_lgbm = optuna.create_study(direction="minimize")
study_lgbm.optimize(objective_lgbm, n_trials=20)
best_lgbm_params = study_lgbm.best_params
print(" Best LGBM:", best_lgbm_params)

# =========================
# Train LGBM 최종
# =========================
lgbm_models = {}
for step in range(1, PREDICT+1):
    X = train_supervised.drop([f"y_{k+1}" for k in range(PREDICT)], axis=1)
    y = train_supervised[f"y_{step}"]
    model = LGBMRegressor(**best_lgbm_params)
    model.fit(X, y)
    lgbm_models[step] = model

# =========================
# LSTM 모델
# =========================
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=7, dropout=0.2):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# LSTM 데이터 준비
def make_lstm_data(df):
    X, y = [], []
    scalers = {}
    for menu_id, group in df.groupby("메뉴ID"):
        vals = group["매출수량"].values
        scaler = MinMaxScaler()
        vals_scaled = scaler.fit_transform(vals.reshape(-1,1)).flatten()
        for i in range(len(vals) - LOOKBACK - PREDICT + 1):
            X.append(vals_scaled[i:i+LOOKBACK])
            y.append(vals_scaled[i+LOOKBACK:i+LOOKBACK+PREDICT])
        scalers[menu_id] = scaler
    return np.array(X), np.array(y), scalers

X_lstm, y_lstm, scalers = make_lstm_data(train)
X_lstm = torch.tensor(X_lstm).float().unsqueeze(-1).to(DEVICE)
y_lstm = torch.tensor(y_lstm).float().to(DEVICE)

# 간단히 Optuna로 hidden_dim 튜닝
def objective_lstm(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    model = MultiOutputLSTM(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    #  전체 대신 일부 샘플만 사용
    n_samples = min(2000, len(X_lstm))
    idx = torch.randperm(len(X_lstm))[:n_samples]
    X_sample, y_sample = X_lstm[idx], y_lstm[idx]
    for epoch in range(3): # 빠른 탐색
        batch_idx = torch.randperm(len(X_sample))
        for i in range(0, len(X_sample), BATCH_SIZE):
            xb, yb = X_sample[batch_idx[i:i+BATCH_SIZE]], y_sample[batch_idx[i:i+BATCH_SIZE]]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        pred = model(X_sample)
    return ((y_sample.cpu().numpy() - pred.cpu().numpy())**2).mean()

print(" LSTM Optuna 튜닝중...")
study_lstm = optuna.create_study(direction="minimize")
study_lstm.optimize(objective_lstm, n_trials=10)
best_lstm_params = study_lstm.best_params
print(" Best LSTM:", best_lstm_params)

# 최종 학습
model_params = {k: v for k, v in best_lstm_params.items() if k != "lr"} # lr 제외
lstm_model = MultiOutputLSTM(**model_params).to(DEVICE)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=best_lstm_params["lr"])
criterion = nn.MSELoss()
for epoch in range(EPOCHS):
    idx = torch.randperm(len(X_lstm))
    for i in range(0, len(X_lstm), BATCH_SIZE):
        batch = idx[i:i+BATCH_SIZE]
        xb, yb = X_lstm[batch], y_lstm[batch]
        pred = lstm_model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
lstm_model.eval()

# =========================
# Prediction
# =========================
def predict_test(test_df, test_prefix):
    test_df = add_features(test_df)
    test_df["메뉴ID"] = le.transform(test_df["영업장명_메뉴명"])
    results = []
    for menu_id, group in test_df.groupby("메뉴ID"):
        group = group.sort_values("영업일자")
        vals = group["매출수량"].values[-LOOKBACK:]
        if len(vals) < LOOKBACK: continue
        feats = group[["요일","월","주말여부","이동평균","이동표준편차","최소값","최대값","증감률"]].iloc[-1]
        # LGBM 예측
        X_input = {
            "메뉴ID": menu_id,
            **{f"lag_{j}": vals[j] for j in range(LOOKBACK)},
            "요일": feats["요일"], "월": feats["월"], "주말여부": feats["주말여부"],
            "이동평균": feats["이동평균"], "이동표준편차": feats["이동표준편차"],
            "최소값": feats["최소값"], "최대값": feats["최대값"], "증감률": feats["증감률"]
        }
        preds_lgbm = []
        vals_tmp = vals.copy()
        for step in range(1, PREDICT+1):
            pred = lgbm_models[step].predict(pd.DataFrame([X_input]))[0]
            preds_lgbm.append(max(pred,0))
            vals_tmp = np.append(vals_tmp[1:], pred)
            for j in range(LOOKBACK):
                X_input[f"lag_{j}"] = vals_tmp[j]
        # LSTM 예측
        scaler = scalers.get(menu_id, MinMaxScaler().fit(vals.reshape(-1,1)))
        vals_scaled = scaler.transform(vals.reshape(-1,1))
        with torch.no_grad():
            pred_scaled = lstm_model(torch.tensor([vals_scaled]).float().to(DEVICE)).cpu().numpy()[0]
        preds_lstm = [max(scaler.inverse_transform([[p]])[0,0],0) for p in pred_scaled]
        # 앙상블
        final_pred = [(a+b)/2 for a,b in zip(preds_lgbm, preds_lstm)]
        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]
        for d, val in zip(pred_dates, final_pred):
            results.append({"영업일자": d, "영업장명_메뉴명": le.inverse_transform([menu_id])[0], "매출수량": val})
    return pd.DataFrame(results)

all_preds = []
for path in test_files:
    test_df = pd.read_csv(path)
    filename = os.path.basename(path)
    test_prefix = re.search(r"(TEST_\d+)", filename).group(1)
    pred_df = predict_test(test_df, test_prefix)
    all_preds.append(pred_df)

full_pred_df = pd.concat(all_preds, ignore_index=True)

# =========================
# Submission
# =========================
def convert_to_submission_format(pred_df, sample_submission):
    pred_dict = dict(zip(zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']), pred_df['매출수량']))
    final_df = sample_submission.copy()
    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)
    return final_df

submission = convert_to_submission_format(full_pred_df, sample_submission)
submission.to_csv("submission_fullstack.csv", index=False, encoding="utf-8-sig")
print(" submission_fullstack.csv 생성 완료")
