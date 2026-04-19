"""
CropGBM Ensemble - 超快速版
3 trials + 3-fold CV + 100 trees
"""
import os, warnings, sys
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

print("=" * 55, flush=True)
print("CropGBM Ensemble: LGB + XGB + CatBoost (Fast Mode)", flush=True)
print("Optuna 3-Trials + 3-Fold CV + Weighted Ensemble", flush=True)
print("=" * 55, flush=True)

# 加载数据
print("\n[1] Loading data...", flush=True)
X_train = pd.read_csv(os.path.join(DATA_DIR, "train.geno"), header=0, index_col=0)
y_train = pd.read_csv(os.path.join(DATA_DIR, "train.phe"), header=0, index_col=0).dropna(axis=0)
X_train = X_train.loc[y_train.index.values, :]
X_valid = pd.read_csv(os.path.join(DATA_DIR, "valid.geno"), header=0, index_col=0)
y_valid = pd.read_csv(os.path.join(DATA_DIR, "valid.phe"), header=0, index_col=0).dropna(axis=0)
X_valid = X_valid.loc[y_valid.index.values, :]
X_test = pd.read_csv(os.path.join(DATA_DIR, "test.geno"), header=0, index_col=0)
print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} SNPs", flush=True)
print(f"  Valid: {X_valid.shape[0]} samples", flush=True)
print(f"  Test: {X_test.shape[0]} samples", flush=True)

N_FOLDS, N_TRIALS, N_EST = 3, 3, 100
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ---- Optuna objectives ----
def lgb_obj(trial):
    p = {
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 48),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 5.0, log=True),
    }
    sc = []
    for ti, vi in kf.split(X_train):
        m = lgb.LGBMRegressor(**p, n_estimators=N_EST, random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(X_train.iloc[ti], y_train.iloc[ti], eval_set=[(X_train.iloc[vi], y_train.iloc[vi])],
              callbacks=[lgb.early_stopping(15, verbose=False)])
        sc.append(np.sqrt(mean_squared_error(y_train.iloc[vi], m.predict(X_train.iloc[vi]))))
    return np.mean(sc)

def xgb_obj(trial):
    p = {
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 5.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 2.0),
    }
    sc = []
    for ti, vi in kf.split(X_train):
        m = xgb.XGBRegressor(**p, n_estimators=N_EST, random_state=42, n_jobs=-1, verbosity=0,
                              early_stopping_rounds=15)
        m.fit(X_train.iloc[ti], y_train.iloc[ti], eval_set=[(X_train.iloc[vi], y_train.iloc[vi])], verbose=False)
        sc.append(np.sqrt(mean_squared_error(y_train.iloc[vi], m.predict(X_train.iloc[vi]))))
    return np.mean(sc)

def cat_obj(trial):
    p = {
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-6, 5.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 30),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    }
    sc = []
    for ti, vi in kf.split(X_train):
        m = CatBoostRegressor(**p, iterations=N_EST, verbose=0, random_seed=42, early_stopping_rounds=15)
        m.fit(X_train.iloc[ti], y_train.iloc[ti], eval_set=(X_train.iloc[vi], y_train.iloc[vi]), verbose=False)
        sc.append(np.sqrt(mean_squared_error(y_train.iloc[vi], m.predict(X_train.iloc[vi]))))
    return np.mean(sc)

# ---- Optuna 调参 ----
print(f"\n[2] Optuna ({N_TRIALS} trials x 3 models)...", flush=True)

print("  Tuning LightGBM...", flush=True)
slgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
slgb.optimize(lgb_obj, n_trials=N_TRIALS, show_progress_bar=False)
print(f"    Best: {slgb.best_value:.5f}", flush=True)

print("  Tuning XGBoost...", flush=True)
sxgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
sxgb.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=False)
print(f"    Best: {sxgb.best_value:.5f}", flush=True)

print("  Tuning CatBoost...", flush=True)
scat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
scat.optimize(cat_obj, n_trials=N_TRIALS, show_progress_bar=False)
print(f"    Best: {scat.best_value:.5f}", flush=True)

bl, bx, bc = slgb.best_params, sxgb.best_params, scat.best_params

# ---- 3折CV + 预测 ----
print(f"\n[3] 3-Fold CV training...", flush=True)
loof, xoof, coof = np.zeros(len(X_train)), np.zeros(len(X_train)), np.zeros(len(X_train))
ltest, xtest, ctest = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))
lf, xf, cf = [], [], []

for i, (ti, vi) in enumerate(kf.split(X_train)):
    print(f"  Fold {i+1}/{N_FOLDS}", end="", flush=True)
    m1 = lgb.LGBMRegressor(**bl, n_estimators=N_EST, random_state=42, n_jobs=-1, verbosity=-1)
    m1.fit(X_train.iloc[ti], y_train.iloc[ti], eval_set=[(X_train.iloc[vi], y_train.iloc[vi])],
           callbacks=[lgb.early_stopping(15, verbose=False)])
    loof[vi] = m1.predict(X_train.iloc[vi])
    ltest += m1.predict(X_test) / N_FOLDS
    lf.append(np.sqrt(mean_squared_error(y_train.iloc[vi], loof[vi])))

    m2 = xgb.XGBRegressor(**bx, n_estimators=N_EST, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=15)
    m2.fit(X_train.iloc[ti], y_train.iloc[ti], eval_set=[(X_train.iloc[vi], y_train.iloc[vi])], verbose=False)
    xoof[vi] = m2.predict(X_train.iloc[vi])
    xtest += m2.predict(X_test) / N_FOLDS
    xf.append(np.sqrt(mean_squared_error(y_train.iloc[vi], xoof[vi])))

    m3 = CatBoostRegressor(**bc, iterations=N_EST, verbose=0, random_seed=42, early_stopping_rounds=15)
    m3.fit(X_train.iloc[ti], y_train.iloc[ti], eval_set=(X_train.iloc[vi], y_train.iloc[vi]), verbose=False)
    coof[vi] = m3.predict(X_train.iloc[vi])
    ctest += m3.predict(X_test) / N_FOLDS
    cf.append(np.sqrt(mean_squared_error(y_train.iloc[vi], coof[vi])))

    print(f"  LGB={lf[-1]:.4f} XGB={xf[-1]:.4f} CAT={cf[-1]:.4f}", flush=True)

# ---- 指标 ----
lr, xr, cr = np.sqrt(mean_squared_error(y_train, loof)), np.sqrt(mean_squared_error(y_train, xoof)), np.sqrt(mean_squared_error(y_train, coof))
lR, xR, cR = r2_score(y_train, loof), r2_score(y_train, xoof), r2_score(y_train, coof)
print(f"\n[4] CV Results:", flush=True)
print(f"  LightGBM RMSE={lr:.5f} R2={lR:.5f}", flush=True)
print(f"  XGBoost  RMSE={xr:.5f} R2={xR:.5f}", flush=True)
print(f"  CatBoost RMSE={cr:.5f} R2={cR:.5f}", flush=True)

# ---- 加权融合 ----
print("\n[5] Weighted Ensemble...", flush=True)
w = 1.0 / np.array([lr, xr, cr])
w = w / w.sum()
print(f"  Weights: LGB={w[0]:.4f} XGB={w[1]:.4f} CAT={w[2]:.4f}", flush=True)
oof_ens = w[0]*loof + w[1]*xoof + w[2]*coof
ens_rmse = np.sqrt(mean_squared_error(y_train, oof_ens))
ens_r2 = r2_score(y_train, oof_ens)
print(f"  Ensemble RMSE={ens_rmse:.5f} R2={ens_r2:.5f}", flush=True)
test_ens = w[0]*ltest + w[1]*xtest + w[2]*ctest

# ---- 保存 ----
print("\n[6] Saving...", flush=True)
pd.DataFrame({'sampleid': X_test.index, 'LightGBM': ltest, 'XGBoost': xtest, 'CatBoost': ctest, 'Ensemble': test_ens}).to_csv(
    os.path.join(OUT, "test_predictions.csv"), index=False)
pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble'],
    'CV_RMSE': [lr, xr, cr, ens_rmse],
    'CV_R2': [lR, xR, cR, ens_r2],
    'Weight': [w[0], w[1], w[2], 1.0]
}).to_csv(os.path.join(OUT, "cv_summary.csv"), index=False)

# ---- 绘图 ----
print("\n[7] Plotting...", flush=True)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
mods, cols = ['LightGBM','XGBoost','CatBoost','Ensemble'], ['#2E86AB','#A23B72','#F18F01','#C73E1D']
rm, r2 = [lr,xr,cr,ens_rmse], [lR,xR,cR,ens_r2]

ax = axes[0,0]
b = ax.bar(mods, rm, color=cols, edgecolor='k', lw=1.2)
for bar, v in zip(b, rm): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('RMSE'); ax.set_title('CV RMSE (Lower Better)', fontweight='bold', fontsize=13)
ax.set_ylim(0, max(rm)*1.25)

ax = axes[0,1]
b = ax.bar(mods, r2, color=cols, edgecolor='k', lw=1.2)
for bar, v in zip(b, r2): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('R2'); ax.set_title('CV R2 (Higher Better)', fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.15)

ax = axes[0,2]
fd = pd.DataFrame({'LightGBM': lf, 'XGBoost': xf, 'CatBoost': cf})
fd.boxplot(ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='#E8E8E8'), medianprops=dict(color='red', lw=2))
ax.set_title('Per-Fold RMSE', fontweight='bold', fontsize=13); ax.set_ylabel('RMSE')

ax = axes[1,0]
ax.scatter(y_train, loof, alpha=0.4, s=10, color='#2E86AB', label=f'LGB R2={lR:.3f}')
ax.scatter(y_train, xoof, alpha=0.4, s=10, color='#A23B72', label=f'XGB R2={xR:.3f}')
ax.scatter(y_train, coof, alpha=0.4, s=10, color='#F18F01', label=f'CAT R2={cR:.3f}')
ax.scatter(y_train, oof_ens, alpha=0.7, s=15, color='#C73E1D', marker='*', label=f'ENS R2={ens_r2:.3f}')
mn, mx = y_train.min().values[0], y_train.max().values[0]
ax.plot([mn,mx],[mn,mx],'k--',lw=1.5)
ax.set_xlabel('True'); ax.set_ylabel('Predicted'); ax.set_title('True vs Predicted (OOF)', fontweight='bold', fontsize=13); ax.legend(fontsize=8)

ax = axes[1,1]
ax.hist(ltest, bins=25, alpha=0.45, color='#2E86AB', label='LightGBM')
ax.hist(xtest, bins=25, alpha=0.45, color='#A23B72', label='XGBoost')
ax.hist(ctest, bins=25, alpha=0.45, color='#F18F01', label='CatBoost')
ax.hist(test_ens, bins=25, alpha=0.75, color='#C73E1D', label='Ensemble', edgecolor='k')
ax.set_xlabel('Predicted'); ax.set_ylabel('Freq'); ax.set_title('Test Prediction Distribution', fontweight='bold', fontsize=13); ax.legend()

ax = axes[1,2]
res = y_train.values.flatten() - oof_ens
ax.hist(res, bins=35, color='#C73E1D', edgecolor='k', alpha=0.85)
ax.axvline(0, color='k', ls='--', lw=2)
ax.set_xlabel('Residual'); ax.set_ylabel('Freq')
ax.set_title(f'Residual: Mean={np.mean(res):.4f}  Std={np.std(res):.4f}', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "ensemble_comparison.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT, "ensemble_comparison.pdf"), bbox_inches='tight')

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for ax, st, nm, cl in zip(axes2, [slgb, sxgb, scat], ['LightGBM','XGBoost','CatBoost'], ['#2E86AB','#A23B72','#F18F01']):
    v = [t.value for t in st.trials]
    ax.plot(v, color=cl, lw=2); ax.axhline(min(v), color='red', ls='--', label=f'Best={min(v):.4f}')
    ax.set_title(f'{nm} Optuna', fontweight='bold'); ax.set_xlabel('Trial'); ax.set_ylabel('CV RMSE'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "optuna_optimization.png"), dpi=150, bbox_inches='tight')

# ---- 汇总 ----
best_s = min(lr, xr, cr)
imp = (best_s - ens_rmse) / best_s * 100

print("\n" + "=" * 55)
print(" SUMMARY")
print("=" * 55)
print(f"  LightGBM   RMSE={lr:.5f}  R2={lR:.5f}  W={w[0]:.4f}")
print(f"  XGBoost    RMSE={xr:.5f}  R2={xR:.5f}  W={w[1]:.4f}")
print(f"  CatBoost   RMSE={cr:.5f}  R2={cR:.5f}  W={w[2]:.4f}")
print("-" * 55)
print(f"  Ensemble   RMSE={ens_rmse:.5f}  R2={ens_r2:.5f}")
print("=" * 55)
if imp > 0: print(f"  Ensemble improves {imp:.2f}% over best single model")
else: print(f"  Ensemble is {-imp:.2f}% worse (single model is best)")
print(f"\n  Output: {OUT}/")
print(f"  Files: ensemble_comparison.png/pdf, optuna_optimization.png, cv_summary.csv, test_predictions.csv")
print("=" * 55)
