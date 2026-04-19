"""
CropGBM Ensemble - Optuna 全量调参版
目标: 最大化提升模型性能
- LightGBM / XGBoost / CatBoost 各 25 trials
- 5-Fold CV
- 300 estimators + early stopping
- 验证集早停 + 最终测试集评估
"""
import os, warnings, time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
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

T_N_FOLDS, T_N_TRIALS, T_N_EST = 5, 25, 300
V_N_EST = 500  # 验证集训练更多树

print("=" * 60)
print("CropGBM Ensemble - Optuna 全量调参版")
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

# 合并训练+验证用于最终模型
X_all = pd.concat([X_train, X_valid], axis=0)
y_all = pd.concat([y_train, y_valid], axis=0)

print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} SNP")
print(f"  验证集: {X_valid.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")
print(f"  全量数据: {X_all.shape[0]} 样本")

kf = KFold(n_splits=T_N_FOLDS, shuffle=True, random_state=42)

# ============ Optuna 目标函数 ============
def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 5.0),
    }
    scores = []
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = lgb.LGBMRegressor(**params, n_estimators=T_N_EST, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        preds = model.predict(X_va)
        scores.append(np.sqrt(mean_squared_error(y_va, preds)))
    return np.mean(scores)

def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
    }
    scores = []
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = xgb.XGBRegressor(**params, n_estimators=T_N_EST, random_state=42, n_jobs=-1, verbosity=0,
                                  early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        scores.append(np.sqrt(mean_squared_error(y_va, preds)))
    return np.mean(scores)

def cat_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 20.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 80),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 20.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
    }
    scores = []
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = CatBoostRegressor(**params, iterations=T_N_EST, verbose=0, random_seed=42,
                                  early_stopping_rounds=30)
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
print(f"    Params: {study_lgb.best_params}")

print("  Tuning XGBoost...", end=" ", flush=True)
t = time.time()
study_xgb = optuna.create_study(direction='minimize', sampler=sampler)
study_xgb.optimize(xgb_objective, n_trials=T_N_TRIALS, show_progress_bar=False)
print(f"Best={study_xgb.best_value:.5f} ({time.time()-t:.0f}s)")
print(f"    Params: {study_xgb.best_params}")

print("  Tuning CatBoost...", end=" ", flush=True)
t = time.time()
study_cat = optuna.create_study(direction='minimize', sampler=sampler)
study_cat.optimize(cat_objective, n_trials=T_N_TRIALS, show_progress_bar=False)
print(f"Best={study_cat.best_value:.5f} ({time.time()-t:.0f}s)")
print(f"    Params: {study_cat.best_params}")

best_lgb = study_lgb.best_params
best_xgb = study_xgb.best_params
best_cat = study_cat.best_params

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
    m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
    lgb_oof[va_idx] = m1.predict(X_va)
    lgb_test += m1.predict(X_test) / T_N_FOLDS
    lgb_fold.append(np.sqrt(mean_squared_error(y_va, lgb_oof[va_idx])))

    # XGBoost
    m2 = xgb.XGBRegressor(**best_xgb, n_estimators=T_N_EST, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=30)
    m2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_oof[va_idx] = m2.predict(X_va)
    xgb_test += m2.predict(X_test) / T_N_FOLDS
    xgb_fold.append(np.sqrt(mean_squared_error(y_va, xgb_oof[va_idx])))

    # CatBoost
    m3 = CatBoostRegressor(**best_cat, iterations=T_N_EST, verbose=0, random_seed=42, early_stopping_rounds=30)
    m3.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    cat_oof[va_idx] = m3.predict(X_va)
    cat_test += m3.predict(X_test) / T_N_FOLDS
    cat_fold.append(np.sqrt(mean_squared_error(y_va, cat_oof[va_idx])))

    print(f"LGB={lgb_fold[-1]:.4f} XGB={xgb_fold[-1]:.4f} CAT={cat_fold[-1]:.4f}")

# ============ 验证集评估 ============
print(f"\n[4] 验证集评估...")
lgb_valid = lgb.LGBMRegressor(**best_lgb, n_estimators=V_N_EST, random_state=42, n_jobs=-1, verbosity=-1)
lgb.LGBMRegressor(**best_lgb, n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1).fit(X_train, y_train)
lgb_valid.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(50, verbose=False)])
lgb_valid_pred = lgb_valid.predict(X_valid)
lgb_valid_rmse = np.sqrt(mean_squared_error(y_valid, lgb_valid_pred))
lgb_valid_r2 = r2_score(y_valid, lgb_valid_pred)

xgb_valid = xgb.XGBRegressor(**best_xgb, n_estimators=V_N_EST, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=50)
xgb_valid.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
xgb_valid_pred = xgb_valid.predict(X_valid)
xgb_valid_rmse = np.sqrt(mean_squared_error(y_valid, xgb_valid_pred))
xgb_valid_r2 = r2_score(y_valid, xgb_valid_pred)

cat_valid = CatBoostRegressor(**best_cat, iterations=V_N_EST, verbose=0, random_seed=42, early_stopping_rounds=50)
cat_valid.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
cat_valid_pred = cat_valid.predict(X_valid)
cat_valid_rmse = np.sqrt(mean_squared_error(y_valid, cat_valid_pred))
cat_valid_r2 = r2_score(y_valid, cat_valid_pred)

# 融合验证集预测
w_v = 1.0 / np.array([lgb_valid_rmse, xgb_valid_rmse, cat_valid_rmse])
w_v = w_v / w_v.sum()
valid_ens_pred = w_v[0]*lgb_valid_pred + w_v[1]*xgb_valid_pred + w_v[2]*cat_valid_pred
valid_ens_rmse = np.sqrt(mean_squared_error(y_valid, valid_ens_pred))
valid_ens_r2 = r2_score(y_valid, valid_ens_pred)

print(f"  LightGBM  Valid RMSE={lgb_valid_rmse:.5f} R2={lgb_valid_r2:.5f}")
print(f"  XGBoost   Valid RMSE={xgb_valid_rmse:.5f} R2={xgb_valid_r2:.5f}")
print(f"  CatBoost  Valid RMSE={cat_valid_rmse:.5f} R2={cat_valid_r2:.5f}")
print(f"  Ensemble  Valid RMSE={valid_ens_rmse:.5f} R2={valid_ens_r2:.5f}")

# ============ CV整体指标 ============
lgb_cv_rmse = np.sqrt(mean_squared_error(y_train, lgb_oof))
xgb_cv_rmse = np.sqrt(mean_squared_error(y_train, xgb_oof))
cat_cv_rmse = np.sqrt(mean_squared_error(y_train, cat_oof))
lgb_cv_r2 = r2_score(y_train, lgb_oof)
xgb_cv_r2 = r2_score(y_train, xgb_oof)
cat_cv_r2 = r2_score(y_train, cat_oof)

# 加权融合
w = 1.0 / np.array([lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse])
w = w / w.sum()
oof_ens = w[0]*lgb_oof + w[1]*xgb_oof + w[2]*cat_oof
ens_cv_rmse = np.sqrt(mean_squared_error(y_train, oof_ens))
ens_cv_r2 = r2_score(y_train, oof_ens)
test_ens = w[0]*lgb_test + w[1]*xgb_test + w[2]*cat_test

print(f"\n[5] 5折CV结果:")
print(f"  LightGBM  CV RMSE={lgb_cv_rmse:.5f} R2={lgb_cv_r2:.5f}")
print(f"  XGBoost   CV RMSE={xgb_cv_rmse:.5f} R2={xgb_cv_r2:.5f}")
print(f"  CatBoost  CV RMSE={cat_cv_rmse:.5f} R2={cat_cv_r2:.5f}")
print(f"  Ensemble  CV RMSE={ens_cv_rmse:.5f} R2={ens_cv_r2:.5f}")
print(f"  权重: LGB={w[0]:.4f} XGB={w[1]:.4f} CAT={w[2]:.4f}")

# ============ 保存 ============
print("\n[6] 保存结果...")
pd.DataFrame({
    'sampleid': X_test.index, 'LightGBM': lgb_test, 'XGBoost': xgb_test,
    'CatBoost': cat_test, 'Ensemble': test_ens
}).to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
pd.DataFrame({
    'sampleid': X_valid.index, 'LightGBM': lgb_valid_pred, 'XGBoost': xgb_valid_pred,
    'CatBoost': cat_valid_pred, 'Ensemble': valid_ens_pred
}).to_csv(os.path.join(OUT, "valid_predictions.csv"), index=False)
pd.DataFrame({
    'Model': ['LightGBM','XGBoost','CatBoost','Ensemble'],
    'CV_RMSE': [lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse, ens_cv_rmse],
    'CV_R2': [lgb_cv_r2, xgb_cv_r2, cat_cv_r2, ens_cv_r2],
    'Valid_RMSE': [lgb_valid_rmse, xgb_valid_rmse, cat_valid_rmse, valid_ens_rmse],
    'Valid_R2': [lgb_valid_r2, xgb_valid_r2, cat_valid_r2, valid_ens_r2],
    'Weight': [w[0], w[1], w[2], 1.0]
}).to_csv(os.path.join(OUT, "cv_summary.csv"), index=False)

# ============ 绘图 ============
print("\n[7] 绘制结果图...")
models = ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble']
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

fig, axes = plt.subplots(2, 4, figsize=(24, 11))

# 1. CV RMSE
ax = axes[0, 0]
vals = [lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse, ens_cv_rmse]
bars = ax.bar(models, vals, color=colors, edgecolor='k', lw=1.2)
for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=12); ax.set_title('5-Fold CV RMSE\n(Lower Better)', fontweight='bold', fontsize=13)
ax.set_ylim(0, max(vals)*1.2)
ax.axhline(ens_cv_rmse, color='purple', ls='--', lw=1.5, alpha=0.7)

# 2. CV R2
ax = axes[0, 1]
vals2 = [lgb_cv_r2, xgb_cv_r2, cat_cv_r2, ens_cv_r2]
bars = ax.bar(models, vals2, color=colors, edgecolor='k', lw=1.2)
for b, v in zip(bars, vals2): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('R²', fontsize=12); ax.set_title('5-Fold CV R²\n(Higher Better)', fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.1)

# 3. 验证集 RMSE vs R2
ax = axes[0, 2]
x_pos = np.arange(len(models))
width = 0.35
vrmse = [lgb_valid_rmse, xgb_valid_rmse, cat_valid_rmse, valid_ens_rmse]
vr2 = [lgb_valid_r2, xgb_valid_r2, cat_valid_r2, valid_ens_r2]
b1 = ax.bar(x_pos - width/2, vrmse, width, label='Valid RMSE', color='steelblue', edgecolor='k')
ax2 = ax.twinx()
b2 = ax2.bar(x_pos + width/2, vr2, width, label='Valid R²', color='coral', edgecolor='k')
ax.set_xticks(x_pos); ax.set_xticklabels(models)
ax.set_ylabel('RMSE', fontsize=11, color='steelblue'); ax2.set_ylabel('R²', fontsize=11, color='coral')
ax.set_title('Validation Set\nRMSE & R²', fontweight='bold', fontsize=13)
ax.legend(loc='upper left'); ax2.legend(loc='upper right')

# 4. 各折RMSE
ax = axes[0, 3]
fd = pd.DataFrame({'LightGBM': lgb_fold, 'XGBoost': xgb_fold, 'CatBoost': cat_fold})
fd.boxplot(ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='#D0D0D0'), medianprops=dict(color='red', lw=2))
ax.set_title('Per-Fold RMSE', fontweight='bold', fontsize=13); ax.set_ylabel('RMSE')

# 5. OOF散点
ax = axes[1, 0]
ax.scatter(y_train, lgb_oof, alpha=0.4, s=12, color='#2196F3', label=f'LGB R²={lgb_cv_r2:.3f}')
ax.scatter(y_train, xgb_oof, alpha=0.4, s=12, color='#FF5722', label=f'XGB R²={xgb_cv_r2:.3f}')
ax.scatter(y_train, cat_oof, alpha=0.4, s=12, color='#4CAF50', label=f'CAT R²={cat_cv_r2:.3f}')
ax.scatter(y_train, oof_ens, alpha=0.7, s=18, color='#9C27B0', marker='*', label=f'ENS R²={ens_cv_r2:.3f}')
mn, mx = y_train.min().values[0], y_train.max().values[0]
ax.plot([mn,mx],[mn,mx],'k--',lw=1.5)
ax.set_xlabel('True', fontsize=12); ax.set_ylabel('Predicted', fontsize=12)
ax.set_title('True vs Predicted (OOF)', fontweight='bold', fontsize=13); ax.legend(fontsize=8)

# 6. 验证集散点
ax = axes[1, 1]
ax.scatter(y_valid, lgb_valid_pred, alpha=0.6, s=40, color='#2196F3', label=f'LGB R²={lgb_valid_r2:.3f}')
ax.scatter(y_valid, xgb_valid_pred, alpha=0.6, s=40, color='#FF5722', label=f'XGB R²={xgb_valid_r2:.3f}')
ax.scatter(y_valid, cat_valid_pred, alpha=0.6, s=40, color='#4CAF50', label=f'CAT R²={cat_valid_r2:.3f}')
ax.scatter(y_valid, valid_ens_pred, alpha=0.8, s=60, color='#9C27B0', marker='*', label=f'ENS R²={valid_ens_r2:.3f}')
mn, mx = y_valid.min().values[0], y_valid.max().values[0]
ax.plot([mn,mx],[mn,mx],'k--',lw=1.5)
ax.set_xlabel('True', fontsize=12); ax.set_ylabel('Predicted', fontsize=12)
ax.set_title('Validation Set True vs Predicted', fontweight='bold', fontsize=13); ax.legend(fontsize=9)

# 7. 测试集预测分布
ax = axes[1, 2]
ax.hist(lgb_test, bins=30, alpha=0.4, color='#2196F3', label='LightGBM')
ax.hist(xgb_test, bins=30, alpha=0.4, color='#FF5722', label='XGBoost')
ax.hist(cat_test, bins=30, alpha=0.4, color='#4CAF50', label='CatBoost')
ax.hist(test_ens, bins=30, alpha=0.75, color='#9C27B0', label='Ensemble', edgecolor='k')
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Freq', fontsize=12)
ax.set_title('Test Prediction Distribution', fontweight='bold', fontsize=13); ax.legend(fontsize=9)

# 8. 残差分布
ax = axes[1, 3]
res = y_train.values.flatten() - oof_ens
ax.hist(res, bins=40, color='#9C27B0', edgecolor='k', alpha=0.85)
ax.axvline(0, color='red', ls='--', lw=2)
ax.set_xlabel('Residual', fontsize=12); ax.set_ylabel('Freq', fontsize=12)
ax.set_title(f'OOF Residual\nMean={np.mean(res):.4f}  Std={np.std(res):.4f}', fontweight='bold', fontsize=13)

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

# 最终汇总
best_single = min(lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse)
improvement = (best_single - ens_cv_rmse) / best_single * 100

total_time = time.time() - t0
print(f"\n总耗时: {total_time/60:.1f} 分钟")
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(f"  LightGBM   CV RMSE={lgb_cv_rmse:.5f}  R2={lgb_cv_r2:.5f}  W={w[0]:.4f}")
print(f"  XGBoost    CV RMSE={xgb_cv_rmse:.5f}  R2={xgb_cv_r2:.5f}  W={w[1]:.4f}")
print(f"  CatBoost   CV RMSE={cat_cv_rmse:.5f}  R2={cat_cv_r2:.5f}  W={w[2]:.4f}")
print("-" * 60)
print(f"  Ensemble   CV RMSE={ens_cv_rmse:.5f}  R2={ens_cv_r2:.5f}")
print("-" * 60)
print(f"  Valid RMSE:  LGB={lgb_valid_rmse:.5f}  XGB={xgb_valid_rmse:.5f}  CAT={cat_valid_rmse:.5f}  ENS={valid_ens_rmse:.5f}")
print(f"  Valid R2:   LGB={lgb_valid_r2:.5f}  XGB={xgb_valid_r2:.5f}  CAT={cat_valid_r2:.5f}  ENS={valid_ens_r2:.5f}")
if improvement > 0:
    print(f"\n  集成提升: {improvement:.2f}% over best single model")
else:
    print(f"\n  集成变化: {-improvement:.2f}%")
print("=" * 60)
