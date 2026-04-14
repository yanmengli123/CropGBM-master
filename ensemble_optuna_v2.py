"""
CropGBM Ensemble - Optuna 增强版
目标: 最大化提升模型性能
- LightGBM / XGBoost / CatBoost 各 50 trials
- 5-Fold CV + Stacking 集成
- 500 estimators + early stopping
- 扩展搜索空间 + 特征筛选
"""
import os, warnings, time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = "./CropGBM-Tutorial-data-main"
OUT = "./ensemble_result"
os.makedirs(OUT, exist_ok=True)

# 增强配置
T_N_FOLDS, T_N_TRIALS, T_N_EST = 5, 50, 500
V_N_EST = 600

print("=" * 60)
print("CropGBM Ensemble - Optuna 增强版")
print(f"  {T_N_TRIALS} trials x 3 models x {T_N_FOLDS}-fold CV")
print("=" * 60)
t0 = time.time()

# ============ 加载数据 ============
print("\n[1] 加载数据...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "train.geno"), header=0, index_col=0)
y_train = pd.read_csv(os.path.join(DATA_DIR, "train.phe"), header=0, index_col=0).dropna(axis=0)
X_train = X_train.loc[y_train.index.values, :]

X_valid = pd.read_csv(os.path.join(DATA_DIR, "valid.geno"), header=0, index_col=0)
y_valid = pd.read_csv(os.path.join(DATA_DIR, "valid.phe"), header=0, index_col=0).dropna(axis=0)
X_valid = X_valid.loc[y_valid.index.values, :]

X_test = pd.read_csv(os.path.join(DATA_DIR, "test.geno"), header=0, index_col=0)

X_all = pd.concat([X_train, X_valid], axis=0)
y_all = pd.concat([y_train, y_valid], axis=0)

print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} SNP")
print(f"  验证集: {X_valid.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")

kf = KFold(n_splits=T_N_FOLDS, shuffle=True, random_state=42)

# ============ Optuna 目标函数 (扩展搜索空间) ============
def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 100),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 30.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 30.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 5.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.0, 10.0),
    }
    scores = []
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = lgb.LGBMRegressor(**params, n_estimators=T_N_EST, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_va)
        scores.append(np.sqrt(mean_squared_error(y_va, preds)))
    return np.mean(scores)

def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 30.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 30.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
    }
    scores = []
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = xgb.XGBRegressor(**params, n_estimators=T_N_EST, random_state=42, n_jobs=-1, verbosity=0,
                                  early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        scores.append(np.sqrt(mean_squared_error(y_va, preds)))
    return np.mean(scores)

def cat_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'depth': trial.suggest_int('depth', 4, 15),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 30.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 3.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 30.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'l1_leaf_reg': trial.suggest_float('l1_leaf_reg', 1e-8, 20.0, log=True),
    }
    scores = []
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = CatBoostRegressor(**params, iterations=T_N_EST, verbose=0, random_seed=42,
                                  early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        preds = model.predict(X_va)
        scores.append(np.sqrt(mean_squared_error(y_va, preds)))
    return np.mean(scores)

# ============ Optuna 调参 ============
print(f"\n[2] Optuna 贝叶斯优化 ({T_N_TRIALS} trials x 3 models)...")

sampler = optuna.samplers.TPESampler(seed=42)

print("  Tuning LightGBM...", end=" ", flush=True)
t = time.time()
study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
study_lgb.optimize(lgb_objective, n_trials=T_N_TRIALS, show_progress_bar=False)
print(f"Best={study_lgb.best_value:.5f} ({time.time()-t:.0f}s)")

print("  Tuning XGBoost...", end=" ", flush=True)
t = time.time()
study_xgb = optuna.create_study(direction='minimize', sampler=sampler)
study_xgb.optimize(xgb_objective, n_trials=T_N_TRIALS, show_progress_bar=False)
print(f"Best={study_xgb.best_value:.5f} ({time.time()-t:.0f}s)")

print("  Tuning CatBoost...", end=" ", flush=True)
t = time.time()
study_cat = optuna.create_study(direction='minimize', sampler=sampler)
study_cat.optimize(cat_objective, n_trials=T_N_TRIALS, show_progress_bar=False)
print(f"Best={study_cat.best_value:.5f} ({time.time()-t:.0f}s)")

best_lgb = study_lgb.best_params
best_xgb = study_xgb.best_params
best_cat = study_cat.best_params

print(f"\n  LGB Best: {study_lgb.best_value:.5f}")
print(f"  XGB Best: {study_xgb.best_value:.5f}")
print(f"  CAT Best: {study_cat.best_value:.5f}")

# ============ 5折CV + OOF预测 ============
print(f"\n[3] 5折CV最终训练 (OOF预测)...")
lgb_oof = np.zeros(len(X_train))
xgb_oof = np.zeros(len(X_train))
cat_oof = np.zeros(len(X_train))
lgb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))
cat_test = np.zeros(len(X_test))
lgb_fold, xgb_fold, cat_fold = [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold+1}/{T_N_FOLDS}", end=" -> ", flush=True)
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    # LightGBM
    m1 = lgb.LGBMRegressor(**best_lgb, n_estimators=T_N_EST, random_state=42, n_jobs=-1, verbosity=-1)
    m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_oof[va_idx] = m1.predict(X_va)
    lgb_test += m1.predict(X_test) / T_N_FOLDS
    lgb_fold.append(np.sqrt(mean_squared_error(y_va, lgb_oof[va_idx])))

    # XGBoost
    m2 = xgb.XGBRegressor(**best_xgb, n_estimators=T_N_EST, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=50)
    m2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_oof[va_idx] = m2.predict(X_va)
    xgb_test += m2.predict(X_test) / T_N_FOLDS
    xgb_fold.append(np.sqrt(mean_squared_error(y_va, xgb_oof[va_idx])))

    # CatBoost
    m3 = CatBoostRegressor(**best_cat, iterations=T_N_EST, verbose=0, random_seed=42, early_stopping_rounds=50)
    m3.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    cat_oof[va_idx] = m3.predict(X_va)
    cat_test += m3.predict(X_test) / T_N_FOLDS
    cat_fold.append(np.sqrt(mean_squared_error(y_va, cat_oof[va_idx])))

    print(f"LGB={lgb_fold[-1]:.4f} XGB={xgb_fold[-1]:.4f} CAT={cat_fold[-1]:.4f}")

# ============ Stacking 集成 ============
print(f"\n[4] Stacking 集成训练...")
# 使用 OOF 预测作为 Stacking meta-features
stack_train = np.column_stack([lgb_oof, xgb_oof, cat_oof])
stack_test = np.column_stack([lgb_test, xgb_test, cat_test])

# Ridge 作为 meta-learner
stack_oof = np.zeros(len(y_train))
stack_fold_test = np.zeros((T_N_FOLDS, len(X_test)))
stack_fold = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    X_tr_stack = stack_train[tr_idx]
    X_va_stack = stack_train[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    meta = Ridge(alpha=1.0)
    meta.fit(X_tr_stack, y_tr)
    stack_oof[va_idx] = meta.predict(X_va_stack).flatten()
    stack_fold_test[fold] = meta.predict(stack_test).flatten()
    stack_fold.append(np.sqrt(mean_squared_error(y_va, stack_oof[va_idx])))

stack_test_final = stack_fold_test.mean(axis=0)

# ============ 验证集评估 ============
print(f"\n[5] 验证集评估...")

# 验证集评估 - 使用训练好的最佳参数在 train 上训练，在 valid 上评估
lgb_valid = lgb.LGBMRegressor(**best_lgb, n_estimators=V_N_EST, random_state=42, n_jobs=-1, verbosity=-1)
lgb_valid.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(50, verbose=False)])
lgb_valid_pred = lgb_valid.predict(X_valid)

xgb_valid = xgb.XGBRegressor(**best_xgb, n_estimators=V_N_EST, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=50)
xgb_valid.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
xgb_valid_pred = xgb_valid.predict(X_valid)

cat_valid = CatBoostRegressor(**best_cat, iterations=V_N_EST, verbose=0, random_seed=42, early_stopping_rounds=50)
cat_valid.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
cat_valid_pred = cat_valid.predict(X_valid)

# Stacking on valid
stack_valid = np.column_stack([lgb_valid_pred, xgb_valid_pred, cat_valid_pred])
meta_valid = Ridge(alpha=1.0)
meta_valid.fit(stack_train, y_train)  # 用全部 train 的 oof 来 fit
stack_valid_pred = meta_valid.predict(stack_valid).flatten()

# 加权融合
w_v = 1.0 / np.array([
    np.sqrt(mean_squared_error(y_valid, lgb_valid_pred)),
    np.sqrt(mean_squared_error(y_valid, xgb_valid_pred)),
    np.sqrt(mean_squared_error(y_valid, cat_valid_pred))
])
w_v = w_v / w_v.sum()
valid_ens_pred = w_v[0]*lgb_valid_pred + w_v[1]*xgb_valid_pred + w_v[2]*cat_valid_pred

# ============ CV整体指标 ============
lgb_cv_rmse = np.sqrt(mean_squared_error(y_train, lgb_oof))
xgb_cv_rmse = np.sqrt(mean_squared_error(y_train, xgb_oof))
cat_cv_rmse = np.sqrt(mean_squared_error(y_train, cat_oof))
stack_cv_rmse = np.sqrt(mean_squared_error(y_train, stack_oof))

lgb_cv_r2 = r2_score(y_train, lgb_oof)
xgb_cv_r2 = r2_score(y_train, xgb_oof)
cat_cv_r2 = r2_score(y_train, cat_oof)
stack_cv_r2 = r2_score(y_train, stack_oof)

# 加权融合 OOF
w = 1.0 / np.array([lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse])
w = w / w.sum()
oof_ens = w[0]*lgb_oof + w[1]*xgb_oof + w[2]*cat_oof
ens_cv_rmse = np.sqrt(mean_squared_error(y_train, oof_ens))
ens_cv_r2 = r2_score(y_train, oof_ens)

# 最终测试预测 - 加权融合 vs Stacking
test_ens = w[0]*lgb_test + w[1]*xgb_test + w[2]*cat_test

# 选择更好的集成方法
if stack_cv_rmse < ens_cv_rmse:
    best_test_pred = stack_test_final
    best_method = "Stacking"
    best_cv_rmse = stack_cv_rmse
    best_cv_r2 = stack_cv_r2
else:
    best_test_pred = test_ens
    best_method = "Weighted"
    best_cv_rmse = ens_cv_rmse
    best_cv_r2 = ens_cv_r2

lgb_valid_rmse = np.sqrt(mean_squared_error(y_valid, lgb_valid_pred))
xgb_valid_rmse = np.sqrt(mean_squared_error(y_valid, xgb_valid_pred))
cat_valid_rmse = np.sqrt(mean_squared_error(y_valid, cat_valid_pred))
stack_valid_rmse = np.sqrt(mean_squared_error(y_valid, stack_valid_pred))
ens_valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_ens_pred))

lgb_valid_r2 = r2_score(y_valid, lgb_valid_pred)
xgb_valid_r2 = r2_score(y_valid, xgb_valid_pred)
cat_valid_r2 = r2_score(y_valid, cat_valid_pred)
stack_valid_r2 = r2_score(y_valid, stack_valid_pred)
ens_valid_r2 = r2_score(y_valid, valid_ens_pred)

print(f"\n[6] 5折CV结果:")
print(f"  LightGBM  CV RMSE={lgb_cv_rmse:.5f} R2={lgb_cv_r2:.5f}")
print(f"  XGBoost   CV RMSE={xgb_cv_rmse:.5f} R2={xgb_cv_r2:.5f}")
print(f"  CatBoost  CV RMSE={cat_cv_rmse:.5f} R2={cat_cv_r2:.5f}")
print(f"  Stacking  CV RMSE={stack_cv_rmse:.5f} R2={stack_cv_r2:.5f}")
print(f"  Weighted  CV RMSE={ens_cv_rmse:.5f} R2={ens_cv_r2:.5f}")
print(f"  Best: {best_method}  CV RMSE={best_cv_rmse:.5f} R2={best_cv_r2:.5f}")
print(f"  权重: LGB={w[0]:.4f} XGB={w[1]:.4f} CAT={w[2]:.4f}")

# ============ 保存 ============
print("\n[7] 保存结果...")
pd.DataFrame({
    'sampleid': X_test.index, 'LightGBM': lgb_test, 'XGBoost': xgb_test,
    'CatBoost': cat_test, 'Weighted_Ensemble': test_ens, 'Stacking_Ensemble': stack_test_final,
    'Best_Ensemble': best_test_pred
}).to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)

pd.DataFrame({
    'sampleid': X_valid.index, 'LightGBM': lgb_valid_pred, 'XGBoost': xgb_valid_pred,
    'CatBoost': cat_valid_pred, 'Weighted_Ensemble': valid_ens_pred, 'Stacking_Ensemble': stack_valid_pred
}).to_csv(os.path.join(OUT, "valid_predictions.csv"), index=False)

pd.DataFrame({
    'Model': ['LightGBM','XGBoost','CatBoost','Stacking','Weighted','Best'],
    'CV_RMSE': [lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse, stack_cv_rmse, ens_cv_rmse, best_cv_rmse],
    'CV_R2': [lgb_cv_r2, xgb_cv_r2, cat_cv_r2, stack_cv_r2, ens_cv_r2, best_cv_r2],
    'Valid_RMSE': [lgb_valid_rmse, xgb_valid_rmse, cat_valid_rmse, stack_valid_rmse, ens_valid_rmse, min(stack_valid_rmse, ens_valid_rmse)],
    'Valid_R2': [lgb_valid_r2, xgb_valid_r2, cat_valid_r2, stack_valid_r2, ens_valid_r2, max(stack_valid_r2, ens_valid_r2)],
    'Weight': [w[0], w[1], w[2], 0.33, 0.67, 1.0]
}).to_csv(os.path.join(OUT, "cv_summary.csv"), index=False)

# ============ 绘图 ============
print("\n[8] 绘制结果图...")
models = ['LightGBM', 'XGBoost', 'CatBoost', 'Stacking', 'Weighted', 'Best']
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#E91E63']

fig, axes = plt.subplots(2, 4, figsize=(24, 11))

# 1. CV RMSE
ax = axes[0, 0]
vals = [lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse, stack_cv_rmse, ens_cv_rmse, best_cv_rmse]
bars = ax.bar(models, vals, color=colors, edgecolor='k', lw=1.2)
for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=12); ax.set_title('5-Fold CV RMSE\n(Lower Better)', fontweight='bold', fontsize=13)
ax.set_ylim(0, max(vals)*1.2)
ax.axhline(min(vals), color='red', ls='--', lw=1.5, alpha=0.7)
ax.tick_params(axis='x', rotation=30)

# 2. CV R2
ax = axes[0, 1]
vals2 = [lgb_cv_r2, xgb_cv_r2, cat_cv_r2, stack_cv_r2, ens_cv_r2, best_cv_r2]
bars = ax.bar(models, vals2, color=colors, edgecolor='k', lw=1.2)
for b, v in zip(bars, vals2): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('R²', fontsize=12); ax.set_title('5-Fold CV R²\n(Higher Better)', fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.1)
ax.tick_params(axis='x', rotation=30)

# 3. 验证集 RMSE vs R2
ax = axes[0, 2]
x_pos = np.arange(len(models))
width = 0.35
vrmse = [lgb_valid_rmse, xgb_valid_rmse, cat_valid_rmse, stack_valid_rmse, ens_valid_rmse, min(stack_valid_rmse, ens_valid_rmse)]
vr2 = [lgb_valid_r2, xgb_valid_r2, cat_valid_r2, stack_valid_r2, ens_valid_r2, max(stack_valid_r2, ens_valid_r2)]
b1 = ax.bar(x_pos - width/2, vrmse, width, label='Valid RMSE', color='steelblue', edgecolor='k')
ax2 = ax.twinx()
b2 = ax2.bar(x_pos + width/2, vr2, width, label='Valid R²', color='coral', edgecolor='k')
ax.set_xticks(x_pos); ax.set_xticklabels(models, rotation=30)
ax.set_ylabel('RMSE', fontsize=11, color='steelblue'); ax2.set_ylabel('R²', fontsize=11, color='coral')
ax.set_title('Validation Set\nRMSE & R²', fontweight='bold', fontsize=13)

# 4. 各折RMSE
ax = axes[0, 3]
fd = pd.DataFrame({'LightGBM': lgb_fold, 'XGBoost': xgb_fold, 'CatBoost': cat_fold, 'Stacking': stack_fold})
fd.boxplot(ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='#D0D0D0'), medianprops=dict(color='red', lw=2))
ax.set_title('Per-Fold RMSE', fontweight='bold', fontsize=13); ax.set_ylabel('RMSE')

# 5. OOF散点
ax = axes[1, 0]
ax.scatter(y_train, lgb_oof, alpha=0.3, s=10, color='#2196F3', label=f'LGB R²={lgb_cv_r2:.3f}')
ax.scatter(y_train, xgb_oof, alpha=0.3, s=10, color='#FF5722', label=f'XGB R²={xgb_cv_r2:.3f}')
ax.scatter(y_train, cat_oof, alpha=0.3, s=10, color='#4CAF50', label=f'CAT R²={cat_cv_r2:.3f}')
ax.scatter(y_train, stack_oof, alpha=0.6, s=14, color='#9C27B0', marker='*', label=f'STK R²={stack_cv_r2:.3f}')
ax.scatter(y_train, oof_ens, alpha=0.6, s=14, color='#FF9800', marker='^', label=f'WGT R²={ens_cv_r2:.3f}')
mn, mx = y_train.min().values[0], y_train.max().values[0]
ax.plot([mn,mx],[mn,mx],'k--',lw=1.5)
ax.set_xlabel('True', fontsize=12); ax.set_ylabel('Predicted', fontsize=12)
ax.set_title('True vs Predicted (OOF)', fontweight='bold', fontsize=13); ax.legend(fontsize=7)

# 6. 验证集散点
ax = axes[1, 1]
ax.scatter(y_valid, lgb_valid_pred, alpha=0.4, s=30, color='#2196F3', label=f'LGB R²={lgb_valid_r2:.3f}')
ax.scatter(y_valid, xgb_valid_pred, alpha=0.4, s=30, color='#FF5722', label=f'XGB R²={xgb_valid_r2:.3f}')
ax.scatter(y_valid, cat_valid_pred, alpha=0.4, s=30, color='#4CAF50', label=f'CAT R²={cat_valid_r2:.3f}')
ax.scatter(y_valid, stack_valid_pred, alpha=0.7, s=50, color='#9C27B0', marker='*', label=f'STK R²={stack_valid_r2:.3f}')
ax.scatter(y_valid, valid_ens_pred, alpha=0.7, s=50, color='#FF9800', marker='^', label=f'WGT R²={ens_valid_r2:.3f}')
mn, mx = y_valid.min().values[0], y_valid.max().values[0]
ax.plot([mn,mx],[mn,mx],'k--',lw=1.5)
ax.set_xlabel('True', fontsize=12); ax.set_ylabel('Predicted', fontsize=12)
ax.set_title('Validation Set True vs Predicted', fontweight='bold', fontsize=13); ax.legend(fontsize=8)

# 7. 测试集预测分布
ax = axes[1, 2]
ax.hist(lgb_test, bins=30, alpha=0.35, color='#2196F3', label='LightGBM')
ax.hist(xgb_test, bins=30, alpha=0.35, color='#FF5722', label='XGBoost')
ax.hist(cat_test, bins=30, alpha=0.35, color='#4CAF50', label='CatBoost')
ax.hist(best_test_pred, bins=30, alpha=0.75, color='#9C27B0', label=f'Best ({best_method})', edgecolor='k')
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Freq', fontsize=12)
ax.set_title('Test Prediction Distribution', fontweight='bold', fontsize=13); ax.legend(fontsize=9)

# 8. 残差分布
ax = axes[1, 3]
res = y_train.values.flatten() - stack_oof
ax.hist(res, bins=40, color='#9C27B0', edgecolor='k', alpha=0.85)
ax.axvline(0, color='red', ls='--', lw=2)
ax.set_xlabel('Residual', fontsize=12); ax.set_ylabel('Freq', fontsize=12)
ax.set_title(f'OOF Residual (Stacking)\nMean={np.mean(res):.4f}  Std={np.std(res):.4f}', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "ensemble_comparison.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT, "ensemble_comparison.pdf"), bbox_inches='tight')
print(f"  saved: ensemble_comparison.png/pdf")

# Optuna 收敛曲线
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for ax, st, nm, cl in zip(axes2, [study_lgb, study_xgb, study_cat],
                            ['LightGBM','XGBoost','CatBoost'], ['#2196F3','#FF5722','#4CAF50']):
    vals = [t.value for t in st.trials]
    ax.plot(vals, color=cl, lw=2, alpha=0.8)
    ax.axhline(min(vals), color='red', ls='--', lw=2, label=f'Best={min(vals):.5f}')
    ax.fill_between(range(len(vals)), vals, alpha=0.2, color=cl)
    ax.set_title(f'{nm} Optuna ({T_N_TRIALS} Trials)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Trial', fontsize=11); ax.set_ylabel('CV RMSE', fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "optuna_optimization.png"), dpi=150, bbox_inches='tight')
print(f"  saved: optuna_optimization.png")

# 最佳参数保存
with open(os.path.join(OUT, "best_params.txt"), 'w') as f:
    f.write("LightGBM Best Params:\n")
    for k, v in best_lgb.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nXGBoost Best Params:\n")
    for k, v in best_xgb.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nCatBoost Best Params:\n")
    for k, v in best_cat.items():
        f.write(f"  {k}: {v}\n")
print(f"  saved: best_params.txt")

# 最终汇总
best_single = min(lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse)
improvement = (best_single - best_cv_rmse) / best_single * 100

total_time = time.time() - t0
print(f"\n总耗时: {total_time/60:.1f} 分钟")
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(f"  LightGBM   CV RMSE={lgb_cv_rmse:.5f}  R2={lgb_cv_r2:.5f}  W={w[0]:.4f}")
print(f"  XGBoost    CV RMSE={xgb_cv_rmse:.5f}  R2={xgb_cv_r2:.5f}  W={w[1]:.4f}")
print(f"  CatBoost   CV RMSE={cat_cv_rmse:.5f}  R2={cat_cv_r2:.5f}  W={w[2]:.4f}")
print("-" * 60)
print(f"  Stacking   CV RMSE={stack_cv_rmse:.5f}  R2={stack_cv_r2:.5f}")
print(f"  Weighted   CV RMSE={ens_cv_rmse:.5f}  R2={ens_cv_r2:.5f}")
print("-" * 60)
print(f"  Best Method: {best_method}")
print(f"  Best CV RMSE={best_cv_rmse:.5f}  R2={best_cv_r2:.5f}")
print(f"  比最优单模型提升: {improvement:.2f}%")
print("=" * 60)