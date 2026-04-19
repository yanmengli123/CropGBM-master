"""
CropGBM Ensemble: LightGBM + XGBoost + CatBoost
Optuna Bayesian Optimization + 5-Fold CV + Weighted Ensemble
精简版: 10 trials + 200 trees 加速
"""
import os
import warnings
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
OUTPUT_DIR = "./ensemble_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("CropGBM Ensemble: LightGBM + XGBoost + CatBoost")
print("Optuna 10 Trials + 5-Fold CV + Weighted Ensemble")
print("=" * 60)

# 加载数据
print("\n[1] 加载数据...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "train.geno"), header=0, index_col=0)
y_train = pd.read_csv(os.path.join(DATA_DIR, "train.phe"), header=0, index_col=0).dropna(axis=0)
X_train = X_train.loc[y_train.index.values, :]

X_valid = pd.read_csv(os.path.join(DATA_DIR, "valid.geno"), header=0, index_col=0)
y_valid = pd.read_csv(os.path.join(DATA_DIR, "valid.phe"), header=0, index_col=0).dropna(axis=0)
X_valid = X_valid.loc[y_valid.index.values, :]

X_test = pd.read_csv(os.path.join(DATA_DIR, "test.geno"), header=0, index_col=0)

print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} SNP")
print(f"  验证集: {X_valid.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
N_TRIALS = 10  # 精简trial数加速
N_ESTIMATORS = 200  # 精简树数量

def lgb_objective(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMRegressor(**params, n_estimators=N_ESTIMATORS,
                                    random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores)

def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 3.0),
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = xgb.XGBRegressor(**params, n_estimators=N_ESTIMATORS,
                                   random_state=42, n_jobs=-1, verbosity=0,
                                   early_stopping_rounds=20)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores)

def cat_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-6, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 40),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostRegressor(**params, iterations=N_ESTIMATORS, verbose=0,
                                  random_seed=42, early_stopping_rounds=20)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores)

# Optuna 调参
print(f"\n[2] Optuna 贝叶斯优化 ({N_TRIALS} trials x 3模型)...")

print("  调参 LightGBM...")
study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"    Best RMSE: {study_lgb.best_value:.5f}")

print("  调参 XGBoost...")
study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"    Best RMSE: {study_xgb.best_value:.5f}")

print("  调参 CatBoost...")
study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_cat.optimize(cat_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"    Best RMSE: {study_cat.best_value:.5f}")

best_lgb = study_lgb.best_params
best_xgb = study_xgb.best_params
best_cat = study_cat.best_params

# 5折CV训练
print(f"\n[3] 5折CV训练 (每折 {N_ESTIMATORS} 棵树)...")

lgb_oof = np.zeros(len(X_train))
xgb_oof = np.zeros(len(X_train))
cat_oof = np.zeros(len(X_train))
lgb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))
cat_test = np.zeros(len(X_test))
lgb_fold, xgb_fold, cat_fold = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=" ")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    lgb_m = lgb.LGBMRegressor(**best_lgb, n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1, verbosity=-1)
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])
    lgb_oof[val_idx] = lgb_m.predict(X_val)
    lgb_test += lgb_m.predict(X_test) / N_FOLDS
    lgb_fold.append(np.sqrt(mean_squared_error(y_val, lgb_oof[val_idx])))

    xgb_m = xgb.XGBRegressor(**best_xgb, n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=20)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_oof[val_idx] = xgb_m.predict(X_val)
    xgb_test += xgb_m.predict(X_test) / N_FOLDS
    xgb_fold.append(np.sqrt(mean_squared_error(y_val, xgb_oof[val_idx])))

    cat_m = CatBoostRegressor(**best_cat, iterations=N_ESTIMATORS, verbose=0, random_seed=42, early_stopping_rounds=20)
    cat_m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    cat_oof[val_idx] = cat_m.predict(X_val)
    cat_test += cat_m.predict(X_test) / N_FOLDS
    cat_fold.append(np.sqrt(mean_squared_error(y_val, cat_oof[val_idx])))

    print(f"LGB={lgb_fold[-1]:.4f} XGB={xgb_fold[-1]:.4f} CAT={cat_fold[-1]:.4f}")

# CV整体
lgb_cv_rmse = np.sqrt(mean_squared_error(y_train, lgb_oof))
xgb_cv_rmse = np.sqrt(mean_squared_error(y_train, xgb_oof))
cat_cv_rmse = np.sqrt(mean_squared_error(y_train, cat_oof))
lgb_cv_r2 = r2_score(y_train, lgb_oof)
xgb_cv_r2 = r2_score(y_train, xgb_oof)
cat_cv_r2 = r2_score(y_train, cat_oof)

print(f"\n[4] CV整体表现:")
print(f"  LightGBM  RMSE: {lgb_cv_rmse:.5f}  R2: {lgb_cv_r2:.5f}")
print(f"  XGBoost   RMSE: {xgb_cv_rmse:.5f}  R2: {xgb_cv_r2:.5f}")
print(f"  CatBoost  RMSE: {cat_cv_rmse:.5f}  R2: {cat_cv_r2:.5f}")

# 加权融合
print("\n[5] 加权融合...")
cv_rmses = np.array([lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse])
weights = (1.0 / cv_rmses)
weights = weights / weights.sum()
print(f"  权重: LGB={weights[0]:.4f} XGB={weights[1]:.4f} CAT={weights[2]:.4f}")

oof_ens = weights[0]*lgb_oof + weights[1]*xgb_oof + weights[2]*cat_oof
ens_rmse = np.sqrt(mean_squared_error(y_train, oof_ens))
ens_r2 = r2_score(y_train, oof_ens)
print(f"  集成   RMSE: {ens_rmse:.5f}  R2: {ens_r2:.5f}")

test_ens = weights[0]*lgb_test + weights[1]*xgb_test + weights[2]*cat_test

# 保存
print("\n[6] 保存结果...")
test_df = pd.DataFrame({
    'sampleid': X_test.index, 'LightGBM': lgb_test, 'XGBoost': xgb_test, 'CatBoost': cat_test, 'Ensemble': test_ens
})
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.predict"), index=False)

summary = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble'],
    'CV_RMSE': [lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse, ens_rmse],
    'CV_R2': [lgb_cv_r2, xgb_cv_r2, cat_cv_r2, ens_r2],
    'Weight': [weights[0], weights[1], weights[2], 1.0]
})
summary.to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"), index=False)

# ===== 绘图 =====
print("\n[7] 绘制对比图...")

models = ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
rmse_vals = [lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse, ens_rmse]
r2_vals = [lgb_cv_r2, xgb_cv_r2, cat_cv_r2, ens_r2]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. CV RMSE
ax = axes[0, 0]
bars = ax.bar(models, rmse_vals, color=colors, edgecolor='black', linewidth=1.2)
for b, v in zip(bars, rmse_vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=12); ax.set_title('5-Fold CV RMSE\n(Lower Better)', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(rmse_vals)*1.25)

# 2. CV R2
ax = axes[0, 1]
bars = ax.bar(models, r2_vals, color=colors, edgecolor='black', linewidth=1.2)
for b, v in zip(bars, r2_vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('R²', fontsize=12); ax.set_title('5-Fold CV R²\n(Higher Better)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.15)

# 3. 各折RMSE
ax = axes[0, 2]
fd = pd.DataFrame({'LightGBM': lgb_fold, 'XGBoost': xgb_fold, 'CatBoost': cat_fold})
fd.boxplot(ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='#E8E8E8'), medianprops=dict(color='red', lw=2))
ax.set_title('Per-Fold RMSE', fontsize=13, fontweight='bold'); ax.set_ylabel('RMSE', fontsize=12)

# 4. OOF散点
ax = axes[1, 0]
ax.scatter(y_train, lgb_oof, alpha=0.4, s=10, color='#2E86AB', label=f'LGB R²={lgb_cv_r2:.3f}')
ax.scatter(y_train, xgb_oof, alpha=0.4, s=10, color='#A23B72', label=f'XGB R²={xgb_cv_r2:.3f}')
ax.scatter(y_train, cat_oof, alpha=0.4, s=10, color='#F18F01', label=f'CAT R²={cat_cv_r2:.3f}')
ax.scatter(y_train, oof_ens, alpha=0.7, s=15, color='#C73E1D', marker='*', label=f'ENS R²={ens_r2:.3f}')
mn, mx = y_train.min().values[0], y_train.max().values[0]
ax.plot([mn,mx],[mn,mx],'k--',lw=1.5); ax.set_xlabel('True', fontsize=12); ax.set_ylabel('Predicted', fontsize=12)
ax.set_title('True vs Predicted (OOF)', fontsize=13, fontweight='bold'); ax.legend(fontsize=8)

# 5. 测试集预测分布
ax = axes[1, 1]
ax.hist(lgb_test, bins=30, alpha=0.45, label='LightGBM', color='#2E86AB')
ax.hist(xgb_test, bins=30, alpha=0.45, label='XGBoost', color='#A23B72')
ax.hist(cat_test, bins=30, alpha=0.45, label='CatBoost', color='#F18F01')
ax.hist(test_ens, bins=30, alpha=0.75, label='Ensemble', color='#C73E1D', edgecolor='black')
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Freq', fontsize=12)
ax.set_title('Test Prediction Distribution', fontsize=13, fontweight='bold'); ax.legend(fontsize=9)

# 6. 残差
ax = axes[1, 2]
res = y_train.values.flatten() - oof_ens
ax.hist(res, bins=40, color='#C73E1D', edgecolor='black', alpha=0.85)
ax.axvline(0, color='black', ls='--', lw=2)
ax.set_xlabel('Residual', fontsize=12); ax.set_ylabel('Freq', fontsize=12)
ax.set_title(f'Ensemble Residual\nMean={np.mean(res):.4f}  Std={np.std(res):.4f}', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ensemble_comparison.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "ensemble_comparison.pdf"), bbox_inches='tight')
print(f"  -> {OUTPUT_DIR}/ensemble_comparison.png/pdf")

# Optuna过程图
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for ax, study, name, color in zip(axes2, [study_lgb, study_xgb, study_cat], ['LightGBM','XGBoost','CatBoost'], ['#2E86AB','#A23B72','#F18F01']):
    vals = [t.value for t in study.trials]
    ax.plot(vals, color=color, lw=1.8, alpha=0.85)
    ax.axhline(min(vals), color='red', ls='--', lw=1.5, label=f'Best: {min(vals):.4f}')
    ax.set_title(f'{name} Optuna ({N_TRIALS} Trials)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=11); ax.set_ylabel('CV RMSE', fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "optuna_optimization.png"), dpi=150, bbox_inches='tight')
print(f"  -> {OUTPUT_DIR}/optuna_optimization.png")

# ===== 汇总 =====
best_single = min(lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse)
improvement = (best_single - ens_rmse) / best_single * 100

print("\n" + "=" * 60)
print("  模型对比汇总")
print("=" * 60)
print(f"{'Model':<15} {'CV_RMSE':>10} {'CV_R2':>10} {'Weight':>10}")
print("-" * 47)
print(f"{'LightGBM':<15} {lgb_cv_rmse:>10.5f} {lgb_cv_r2:>10.5f} {weights[0]:>10.4f}")
print(f"{'XGBoost':<15} {xgb_cv_rmse:>10.5f} {xgb_cv_r2:>10.5f} {weights[1]:>10.4f}")
print(f"{'CatBoost':<15} {cat_cv_rmse:>10.5f} {cat_cv_r2:>10.5f} {weights[2]:>10.4f}")
print("-" * 47)
print(f"{'Ensemble':<15} {ens_rmse:>10.5f} {ens_r2:>10.5f} {'1.0000':>10}")
print("=" * 60)
print(f"\n最优单模型: {best_single:.5f}")
print(f"集成模型:   {ens_rmse:.5f}")
if improvement > 0:
    print(f"集成提升:   {improvement:.2f}%")
else:
    print(f"集成下降:   {-improvement:.2f}% (单模型已够好)")
print(f"\n最优参数:")
print(f"  LightGBM: {best_lgb}")
print(f"  XGBoost:  {best_xgb}")
print(f"  CatBoost: {best_cat}")
print("=" * 60)
