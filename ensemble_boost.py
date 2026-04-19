"""
CropGBM Ensemble: LightGBM + XGBoost + CatBoost
Optuna Bayesian Optimization + 5-Fold CV + Weighted Ensemble
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============ 数据路径 ============
DATA_DIR = "./CropGBM-Tutorial-data-main"
TRAIN_GENO = os.path.join(DATA_DIR, "train.geno")
TRAIN_PHE = os.path.join(DATA_DIR, "train.phe")
VALID_GENO = os.path.join(DATA_DIR, "valid.geno")
VALID_PHE = os.path.join(DATA_DIR, "valid.phe")
TEST_GENO = os.path.join(DATA_DIR, "test.geno")

OUTPUT_DIR = "./ensemble_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("CropGBM Ensemble: LightGBM + XGBoost + CatBoost")
print("Optuna Bayesian Optimization + 5-Fold CV + Weighted Ensemble")
print("=" * 60)

# ============ 加载数据 ============
print("\n[1] 加载数据...")
X_train = pd.read_csv(TRAIN_GENO, header=0, index_col=0)
y_train = pd.read_csv(TRAIN_PHE, header=0, index_col=0).dropna(axis=0)
X_train = X_train.loc[y_train.index.values, :]

X_valid = pd.read_csv(VALID_GENO, header=0, index_col=0)
y_valid = pd.read_csv(VALID_PHE, header=0, index_col=0).dropna(axis=0)
X_valid = X_valid.loc[y_valid.index.values, :]

X_test = pd.read_csv(TEST_GENO, header=0, index_col=0)

print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} SNP位点")
print(f"  验证集: {X_valid.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")

# ============ 5折交叉验证准备 ============
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ============ Optuna 目标函数 ============
def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 5, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMRegressor(**params, n_estimators=500,
                                    random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores)

def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = xgb.XGBRegressor(**params, n_estimators=500,
                                   random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  verbose=False)
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores)

def cat_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostRegressor(**params, iterations=500, verbose=0,
                                   random_seed=42, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores)

# ============ Optuna 调参 ============
print("\n[2] Optuna 贝叶斯优化调参 (每个模型 30 trials)...")

study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(lgb_objective, n_trials=30, show_progress_bar=False)
print(f"  LightGBM best RMSE: {study_lgb.best_value:.5f}")

study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_objective, n_trials=30, show_progress_bar=False)
print(f"  XGBoost  best RMSE: {study_xgb.best_value:.5f}")

study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_cat.optimize(cat_objective, n_trials=30, show_progress_bar=False)
print(f"  CatBoost  best RMSE: {study_cat.best_value:.5f}")

# ============ 获取最优参数 ============
best_lgb = study_lgb.best_params
best_xgb = study_xgb.best_params
best_cat = study_cat.best_params

# ============ 5折CV训练 & 预测 ============
print("\n[3] 5折交叉验证训练...")

lgb_oof = np.zeros(len(X_train))
xgb_oof = np.zeros(len(X_train))
cat_oof = np.zeros(len(X_train))

lgb_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))
cat_test_preds = np.zeros(len(X_test))

lgb_fold_scores = []
xgb_fold_scores = []
cat_fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=" ")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**best_lgb, n_estimators=500,
                                    random_state=42, n_jobs=-1, verbosity=-1)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_test_preds += lgb_model.predict(X_test) / N_FOLDS
    lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_oof[val_idx]))
    lgb_fold_scores.append(lgb_rmse)

    # XGBoost
    xgb_model = xgb.XGBRegressor(**best_xgb, n_estimators=500,
                                   random_state=42, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_test_preds += xgb_model.predict(X_test) / N_FOLDS
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_oof[val_idx]))
    xgb_fold_scores.append(xgb_rmse)

    # CatBoost
    cat_model = CatBoostRegressor(**best_cat, iterations=500, verbose=0,
                                  random_seed=42, early_stopping_rounds=30)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    cat_oof[val_idx] = cat_model.predict(X_val)
    cat_test_preds += cat_model.predict(X_test) / N_FOLDS
    cat_rmse = np.sqrt(mean_squared_error(y_val, cat_oof[val_idx]))
    cat_fold_scores.append(cat_rmse)

    print(f"LGB={lgb_rmse:.4f} XGB={xgb_rmse:.4f} CAT={cat_rmse:.4f}")

# ============ CV整体表现 ============
print("\n[4] 5折CV整体表现...")
lgb_cv_rmse = np.sqrt(mean_squared_error(y_train, lgb_oof))
xgb_cv_rmse = np.sqrt(mean_squared_error(y_train, xgb_oof))
cat_cv_rmse = np.sqrt(mean_squared_error(y_train, cat_oof))

lgb_cv_r2 = r2_score(y_train, lgb_oof)
xgb_cv_r2 = r2_score(y_train, xgb_oof)
cat_cv_r2 = r2_score(y_train, cat_oof)

print(f"  LightGBM  CV-RMSE: {lgb_cv_rmse:.5f}  CV-R2: {lgb_cv_r2:.5f}")
print(f"  XGBoost   CV-RMSE: {xgb_cv_rmse:.5f}  CV-R2: {xgb_cv_r2:.5f}")
print(f"  CatBoost  CV-RMSE: {cat_cv_rmse:.5f}  CV-R2: {cat_cv_r2:.5f}")

# ============ 加权融合 ============
print("\n[5] 基于CV表现的加权融合...")
cv_rmses = np.array([lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse])
# 权重 = 1/RMSE，归一化
weights = 1.0 / cv_rmses
weights = weights / weights.sum()

print(f"  权重: LightGBM={weights[0]:.4f}  XGBoost={weights[1]:.4f}  CatBoost={weights[2]:.4f}")

# OOF融合
oof_ensemble = weights[0]*lgb_oof + weights[1]*xgb_oof + weights[2]*cat_oof
ens_cv_rmse = np.sqrt(mean_squared_error(y_train, oof_ensemble))
ens_cv_r2 = r2_score(y_train, oof_ensemble)
print(f"  加权融合 CV-RMSE: {ens_cv_rmse:.5f}  CV-R2: {ens_cv_r2:.5f}")

# 测试集融合
test_ensemble = weights[0]*lgb_test_preds + weights[1]*xgb_test_preds + weights[2]*cat_test_preds

# ============ 保存预测结果 ============
print("\n[6] 保存结果...")
test_df = pd.DataFrame({
    'sampleid': X_test.index,
    'LightGBM': lgb_test_preds,
    'XGBoost': xgb_test_preds,
    'CatBoost': cat_test_preds,
    'Ensemble': test_ensemble
})
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.predict"), index=False)

# ============ 绘制图表 ============
print("\n[7] 绘制对比图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# ---- 图1: CV RMSE 对比 ----
ax1 = axes[0, 0]
models = ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble']
rmse_vals = [lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse, ens_cv_rmse]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax1.bar(models, rmse_vals, color=colors, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, rmse_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('5-Fold CV RMSE Comparison\n(Lower is Better)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(rmse_vals) * 1.2)

# ---- 图2: CV R² 对比 ----
ax2 = axes[0, 1]
r2_vals = [lgb_cv_r2, xgb_cv_r2, cat_cv_r2, ens_cv_r2]
bars2 = ax2.bar(models, r2_vals, color=colors, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars2, r2_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylabel('R²', fontsize=12)
ax2.set_title('5-Fold CV R² Comparison\n(Higher is Better)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.1)

# ---- 图3: 各折RMSE箱线图 ----
ax3 = axes[0, 2]
fold_data = pd.DataFrame({
    'LightGBM': lgb_fold_scores,
    'XGBoost': xgb_fold_scores,
    'CatBoost': cat_fold_scores
})
fold_data.boxplot(ax=ax3, grid=False, patch_artist=True,
                  boxprops=dict(facecolor='#E8E8E8'),
                  medianprops=dict(color='red', linewidth=2))
ax3.set_title('Per-Fold RMSE Distribution', fontsize=13, fontweight='bold')
ax3.set_ylabel('RMSE', fontsize=12)

# ---- 图4: OOF 真实值 vs 预测值 散点图 ----
ax4 = axes[1, 0]
ax4.scatter(y_train, lgb_oof, alpha=0.5, s=15, label=f'LightGBM (R²={lgb_cv_r2:.3f})', color='#2E86AB')
ax4.scatter(y_train, xgb_oof, alpha=0.5, s=15, label=f'XGBoost (R²={xgb_cv_r2:.3f})', color='#A23B72')
ax4.scatter(y_train, cat_oof, alpha=0.5, s=15, label=f'CatBoost (R²={cat_cv_r2:.3f})', color='#F18F01')
ax4.scatter(y_train, oof_ensemble, alpha=0.7, s=20, label=f'Ensemble (R²={ens_cv_r2:.3f})', color='#C73E1D', marker='*')
min_val, max_val = y_train.min().values[0], y_train.max().values[0]
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='y=x')
ax4.set_xlabel('True Phenotype', fontsize=12)
ax4.set_ylabel('Predicted (OOF)', fontsize=12)
ax4.set_title('True vs Predicted (OOF)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=8, loc='upper left')

# ---- 图5: 预测值分布对比 ----
ax5 = axes[1, 1]
ax5.hist(lgb_test_preds, bins=30, alpha=0.5, label='LightGBM', color='#2E86AB')
ax5.hist(xgb_test_preds, bins=30, alpha=0.5, label='XGBoost', color='#A23B72')
ax5.hist(cat_test_preds, bins=30, alpha=0.5, label='CatBoost', color='#F18F01')
ax5.hist(test_ensemble, bins=30, alpha=0.7, label='Ensemble', color='#C73E1D', edgecolor='black')
ax5.set_xlabel('Predicted Phenotype', fontsize=12)
ax5.set_ylabel('Frequency', fontsize=12)
ax5.set_title('Test Set Prediction Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)

# ---- 图6: 残差分布 ----
ax6 = axes[1, 2]
residuals = y_train.values.flatten() - oof_ensemble
ax6.hist(residuals, bins=40, color='#C73E1D', edgecolor='black', alpha=0.8)
ax6.axvline(x=0, color='black', linestyle='--', lw=2)
ax6.set_xlabel('Residual (True - Ensemble)', fontsize=12)
ax6.set_ylabel('Frequency', fontsize=12)
ax6.set_title(f'Ensemble Residual Distribution\nMean={np.mean(residuals):.4f}  Std={np.std(residuals):.4f}',
              fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ensemble_comparison.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "ensemble_comparison.pdf"), bbox_inches='tight')
print(f"  保存至: {OUTPUT_DIR}/ensemble_comparison.png/pdf")

# ---- 额外: Optuna 优化过程图 ----
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for ax, study, name, color in zip(axes2,
                                    [study_lgb, study_xgb, study_cat],
                                    ['LightGBM', 'XGBoost', 'CatBoost'],
                                    ['#2E86AB', '#A23B72', '#F18F01']):
    vals = [trial.value for trial in study.trials]
    ax.plot(vals, color=color, lw=1.5, alpha=0.8)
    ax.axhline(y=min(vals), color='red', linestyle='--', lw=1.5, label=f'Best: {min(vals):.4f}')
    ax.set_title(f'{name} Optuna Optimization\n(30 Trials)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=11)
    ax.set_ylabel('CV RMSE', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "optuna_optimization.png"), dpi=150, bbox_inches='tight')
print(f"  保存至: {OUTPUT_DIR}/optuna_optimization.png")

# ---- 保存汇总表 ----
summary = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble (Weighted)'],
    'CV_RMSE': [lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse, ens_cv_rmse],
    'CV_R2': [lgb_cv_r2, xgb_cv_r2, cat_cv_r2, ens_cv_r2],
    'Weight': [weights[0], weights[1], weights[2], 1.0]
})
summary.to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"), index=False)
print(f"  保存至: {OUTPUT_DIR}/cv_summary.csv")

# ============ 最终输出 ============
print("\n" + "=" * 60)
print("  模型对比汇总")
print("=" * 60)
print(f"{'Model':<20} {'CV_RMSE':>10} {'CV_R2':>10} {'Weight':>10}")
print("-" * 52)
print(f"{'LightGBM':<20} {lgb_cv_rmse:>10.5f} {lgb_cv_r2:>10.5f} {weights[0]:>10.4f}")
print(f"{'XGBoost':<20} {xgb_cv_rmse:>10.5f} {xgb_cv_r2:>10.5f} {weights[1]:>10.4f}")
print(f"{'CatBoost':<20} {cat_cv_rmse:>10.5f} {cat_cv_r2:>10.5f} {weights[2]:>10.4f}")
print("-" * 52)
print(f"{'Ensemble':<20} {ens_cv_rmse:>10.5f} {ens_cv_r2:>10.5f} {'1.0000':>10}")
print("=" * 60)
best_single = min(lgb_cv_rmse, xgb_cv_rmse, cat_cv_rmse)
improvement = (best_single - ens_cv_rmse) / best_single * 100
print(f"\n最优单模型 RMSE: {best_single:.5f}")
print(f"集成模型 RMSE: {ens_cv_rmse:.5f}")
print(f"集成提升: {improvement:.2f}%")
print("\n输出目录:", os.path.abspath(OUTPUT_DIR))
print("=" * 60)
