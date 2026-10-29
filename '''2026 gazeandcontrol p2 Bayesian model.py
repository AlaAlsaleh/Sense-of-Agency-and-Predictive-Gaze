'''2026 gazeandcontrol p2 Bayesian model analysis'''




# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import pymc as pm
import scipy as sp
from scipy.stats import foldnorm
import pytensor.tensor as pt
from sklearn.metrics import roc_curve, auc







print(f"Running on PyMC v{pm.__version__}")


#%%
all_data = pd.read_csv('all_participants_sacccount.csv')

# omitting zero and nagtive mov_speed's trials -> invalid trials
all_data = all_data.groupby(['pt','b','trial']) \
                   .filter(lambda x: (x['mov_speed'] > 0).all())

# Keep only trials where mov_length is non-negative
all_data = all_data[all_data['mov_length'] >= 0].copy()

# Optional: reset the index
all_data.reset_index(drop=True, inplace=True)

# Add mean saccade velocity
all_data['sacc_velocity'] = all_data['sacc_amp'] / (all_data['sacc_end_time'] - all_data['sacc_start_time'])


all_data = all_data[np.isfinite(all_data['sacc_velocity'])].copy()
all_data.reset_index(drop=True, inplace=True)


# Calculating the rolling uncertainty around GE
all_data["GE_rolling_std"] = (
    all_data.groupby(["pt", "b"])["gaze_e"]
    .transform(lambda x: x.rolling(window=5, min_periods=2).std())
)

# Drop NaN rows introduced by rolling (first trials per participant/block)
all_data = all_data.dropna(subset=["GE_rolling_std"]).reset_index(drop=True)



# Check
print(all_data.shape)



#%%
# Prepare response coding 0/1 from ctrl (±1)
ctrl_01 = (all_data['ctrl'].values + 1) / 2  # -1->0, 1->1
all_data["ctrl"] = ctrl_01 

# Unique participants and blocks
participants = all_data['pt'].values
blocks = all_data['b'].values
unique_participants = np.unique(participants)
unique_blocks = np.unique(blocks)

# Map participant and block to integer indices
pt_idx = np.array([np.where(unique_participants == p)[0][0] for p in participants])
block_idx = np.array([np.where(unique_blocks == b)[0][0] for b in blocks])

# Identify participant groups based on blocks they appear in
pt_block_map = all_data.groupby('pt')['b'].unique()

n_group = 2

# Group 1: seen block 1 or 2; Group 2: only seen block 3
group_1_pts = pt_block_map[pt_block_map.apply(lambda bs: any(b in [1, 2] for b in bs))].index
group_2_pts = pt_block_map[pt_block_map.apply(lambda bs: all(b == 3 for b in bs))].index

# Assign group indices (0 for Group 1, 1 for Group 2)
group_idx = np.array([
    0 if pt in group_1_pts else 1 for pt in participants
])



draws = 5000
tune = 2000
target_accept = 0.92

coords = {"observation": all_data.index.values}




#%%
############### Adding pervious ctrl response and previous GE as Predictors ###############
###########################################################################################

# Sorting by participant and trial to ensure correct lagging
# all_data = all_data.sort_values(["pt", "trial"]).reset_index(drop=True)
all_data = all_data.sort_values(["pt", "b", "trial"]).reset_index(drop=True)


# Lagged absolute gaze error (previous trial per participant)
# all_data["GE_lag1"] = all_data.groupby("pt")["gaze_e"].shift(1).abs()
all_data["GE_lag1"] = all_data.groupby(["pt", "b"])["gaze_e"].shift(1).abs()

# Lagged ctrl (previous trial response)
# all_data["ctrl_lag1"] = all_data.groupby("pt")["ctrl"].shift(1)
all_data["ctrl_lag1"] = all_data.groupby(["pt", "b"])["ctrl"].shift(1)



#%%
# ============================================================
# Standardize all predictors
# ============================================================

def standardize(col):
    return (col - col.mean()) / col.std()

def standardize_abs(col):
    return (col.abs() - col.abs().mean()) / col.abs().std()

all_data["sacc_onset_diff"] = all_data["sacc_start_time"] - all_data["mov_onset_time"]
all_data["sacc_onset_diff_bin"] = (all_data["sacc_onset_diff"] < 0).astype(int)

predictors = {
    "trial"               : standardize(all_data["trial"]),
    "gain"                : standardize(all_data["gain"]),
    "ge_abs"              : standardize_abs(all_data["gaze_e"]),
    "ge_signed"           : all_data["gaze_e"] / all_data["gaze_e"].std(),
    "ge_orig"             : standardize(all_data["gaze_e_orig"]),
    "dist_cursor_amp"     : standardize(all_data["dist_cursor_amp"]),
    "orig_cursor_amp"     : standardize(all_data["orig_cursor_amp"]),
    "sacc_amp"            : standardize(all_data["sacc_amp"]),
    "mov_length"          : standardize(all_data["mov_length"]),
    "mov_speed"           : standardize(all_data["mov_speed"]),
    "mov_onset"           : standardize(all_data["mov_onset_time"]),
    "n_sacc"              : standardize(all_data["n_sacc"]),
    "sacc_onset_diff_bin" : all_data["sacc_onset_diff_bin"].astype(float),
    "sacc_velocity"       : standardize(all_data["sacc_velocity"]),
    "GE_rolling_std"      : standardize(all_data["GE_rolling_std"])
}


#%%
# ============================================================
# Single predictor model specs — 14 models
# ============================================================

model_specs_single = {
    "M01_ge_abs"              : ["ge_abs"],
    "M02_ge_signed"           : ["ge_signed"],
    "M03_gain"                : ["gain"],
    "M04_trial"               : ["trial"],
    "M05_sacc_amp"            : ["sacc_amp"],
    "M06_mov_length"          : ["mov_length"],
    "M07_mov_speed"           : ["mov_speed"],
    "M08_n_sacc"              : ["n_sacc"],
    "M09_sacc_velocity"       : ["sacc_velocity"],
    "M10_dist_cursor"         : ["dist_cursor_amp"],
    "M11_orig_cursor"         : ["orig_cursor_amp"],
    "M12_sacc_onset_diff_bin" : ["sacc_onset_diff_bin"],
    "M13_GE_rolling_std"      : ["GE_rolling_std"],
}

# ============================================================
# Helper: build and sample one hierarchical model
# ============================================================

def build_and_sample_model(model_name, predictor_names,
                            draws=5000, tune=2000, target_accept=0.92):

    pred_data = {k: predictors[k].values for k in predictor_names}

    with pm.Model(coords=coords) as model:

        X = {k: pm.Data(k, v, dims="observation") for k, v in pred_data.items()}

        # Intercept
        mu_a    = pm.Normal('mu_a',   mu=0, sigma=1, shape=len(unique_blocks))
        sigma_a = pm.Gamma('sigma_a', alpha=3, beta=3, shape=len(unique_blocks))
        a_off   = pm.Normal('a_off',  mu=0, sigma=1,
                            shape=(len(unique_participants), len(unique_blocks)))
        a       = pm.Deterministic('a', mu_a + a_off * sigma_a)

        # Slopes
        slopes = {}
        for pname in predictor_names:
            mu_b    = pm.Normal(f'mu_b_{pname}',   0, 1, shape=len(unique_blocks))
            sigma_b = pm.Gamma(f'sigma_b_{pname}', alpha=3, beta=3,
                               shape=len(unique_blocks))
            b_off   = pm.Normal(f'b_{pname}_off',  mu=0, sigma=1,
                                shape=(len(unique_participants), len(unique_blocks)))
            b       = pm.Deterministic(f'b_{pname}', mu_b + b_off * sigma_b)
            slopes[pname] = b

        # Linear predictor
        mu_y = a[pt_idx, block_idx]
        for pname in predictor_names:
            mu_y = mu_y + slopes[pname][pt_idx, block_idx] * X[pname]

        p = pm.Deterministic("p", pm.math.invlogit(mu_y), dims="observation")
        y = pm.Bernoulli('y', p=p, observed=ctrl_01)

        trace = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept,
            return_inferencedata=True, chains=4, cores=4,
            nuts_sampler="numpyro",
            idata_kwargs={"log_likelihood": True}
        )

    return model, trace


# ============================================================
# Run all single predictor models
# ============================================================

traces_single  = {}
summaries      = {}
loo_results    = {}
waic_results   = {}
failed_single  = []

for model_name, pred_list in model_specs_single.items():
    print(f"\n{'='*50}")
    print(f"Fitting: {model_name} | Predictor: {pred_list[0]}")

    try:
        _, trace = build_and_sample_model(model_name, pred_list)
        traces_single[model_name] = trace

        # Save trace
        az.to_netcdf(trace, f"summary_stats/{model_name}.nc")

        # Summary stats
        pname = pred_list[0]
        summary = az.summary(
            trace,
            var_names=["mu_a", f"mu_b_{pname}", f"sigma_b_{pname}", "sigma_a"],
            hdi_prob=0.94
        )
        summaries[model_name] = summary
        summary.to_excel(f"summary_stats/{model_name}_summary.xlsx", index=True)
        print(f"\nSummary — {model_name}:")
        print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]])

        # LOO
        loo = az.loo(trace, scale="log")
        loo_results[model_name] = loo
        print(f"\nLOO — {model_name}:")
        print(loo)

        # WAIC
        waic = az.waic(trace, scale="log")
        waic_results[model_name] = waic
        print(f"\nWAIC — {model_name}:")
        print(waic)

    except Exception as e:
        print(f"  FAILED: {model_name} — {e}")
        failed_single.append(model_name)

print(f"\nFailed models: {failed_single}")

# ============================================================
# LOO model comparison across all 13
# ============================================================

comparison_single = az.compare(
    traces_single, ic="loo", scale="deviance"
)
print("\nFull LOO Model Comparison (ranked):")
print(comparison_single[["elpd_loo", "p_loo", "elpd_diff",
                           "weight", "se", "dse"]].to_string())
comparison_single.to_excel(
    "summary_stats/loo_single_predictors_comparison.xlsx", index=True
)

# ============================================================
# Summary table: slope HDI across all models
# ============================================================

print("\n" + "="*60)
print("SLOPE SUMMARY — mu_b per model (HDI 94%)")
print("="*60)
rows = []
for model_name, trace in traces_single.items():
    pname = model_specs_single[model_name][0]
    s = az.summary(trace, var_names=[f"mu_b_{pname}"], hdi_prob=0.94)
    for i, row in s.iterrows():
        rows.append({
            "model"    : model_name,
            "predictor": pname,
            "block"    : i,
            "mean"     : row["mean"],
            "sd"       : row["sd"],
            "hdi_3%"   : row["hdi_3%"],
            "hdi_97%"  : row["hdi_97%"],
            "r_hat"    : row["r_hat"],
            "elpd_loo" : loo_results[model_name].elpd_loo
                         if model_name in loo_results else np.nan
        })

slope_summary_df = pd.DataFrame(rows)
slope_summary_df.to_excel(
    "summary_stats/slope_summary_all_single_models.xlsx", index=False
)
print(slope_summary_df.to_string())

# ============================================================
# Plots
# ============================================================

# LOO comparison plot
az.plot_compare(comparison_single)
plt.title("LOO Comparison: Single Predictor Models (M01–M13)")
plt.tight_layout()
plt.savefig("remote_plots/loo_single_predictors.png", dpi=150, bbox_inches='tight')
plt.show()

# Slope forest plot per model
fig, axes = plt.subplots(1, len(model_specs_single),
                          figsize=(3*len(model_specs_single), 5),
                          sharey=False)

for ax, (model_name, pred_list) in zip(axes, model_specs_single.items()):
    if model_name not in traces_single:
        continue
    pname = pred_list[0]
    trace = traces_single[model_name]
    s = az.summary(trace, var_names=[f"mu_b_{pname}"], hdi_prob=0.94)

    means  = s["mean"].values
    lo     = s["hdi_3%"].values
    hi     = s["hdi_97%"].values
    blocks = [f"B{b}" for b in unique_blocks]

    ax.errorbar(means, np.arange(len(blocks)),
                xerr=[means - lo, hi - means],
                fmt='o', color='steelblue', capsize=4)
    ax.axvline(0, linestyle='--', color='gray', alpha=0.7)
    ax.set_yticks(np.arange(len(blocks)))
    ax.set_yticklabels(blocks)
    ax.set_title(pname, fontsize=8)
    ax.set_xlabel("slope", fontsize=7)

plt.suptitle("Posterior slopes (mu_b) per predictor per block\n94% HDI",
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig("remote_plots/slope_forest_single_predictors.png",
            dpi=150, bbox_inches='tight')
plt.show()





#%%

# ============================================================
# Helper: build and sample one hierarchical model
# ============================================================

def build_and_sample_model(model_name, predictor_names,
                            draws=5000, tune=2000, target_accept=0.92):

    pred_data = {k: predictors[k].values for k in predictor_names}

    with pm.Model(coords=coords) as model:

        X = {k: pm.Data(k, v, dims="observation") for k, v in pred_data.items()}

        # Intercept
        mu_a    = pm.Normal('mu_a',   mu=0, sigma=1, shape=len(unique_blocks))
        sigma_a = pm.Gamma('sigma_a', alpha=3, beta=3, shape=len(unique_blocks))
        a_off   = pm.Normal('a_off',  mu=0, sigma=1,
                            shape=(len(unique_participants), len(unique_blocks)))
        a       = pm.Deterministic('a', mu_a + a_off * sigma_a)

        # Slopes
        slopes = {}
        for pname in predictor_names:
            mu_b    = pm.Normal(f'mu_b_{pname}',   0, 1, shape=len(unique_blocks))
            sigma_b = pm.Gamma(f'sigma_b_{pname}', alpha=3, beta=3,
                               shape=len(unique_blocks))
            b_off   = pm.Normal(f'b_{pname}_off',  mu=0, sigma=1,
                                shape=(len(unique_participants), len(unique_blocks)))
            b       = pm.Deterministic(f'b_{pname}', mu_b + b_off * sigma_b)
            slopes[pname] = b

        # Linear predictor
        mu_y = a[pt_idx, block_idx]
        for pname in predictor_names:
            mu_y = mu_y + slopes[pname][pt_idx, block_idx] * X[pname]

        p = pm.Deterministic("p", pm.math.invlogit(mu_y), dims="observation")
        y = pm.Bernoulli('y', p=p, observed=ctrl_01)

        trace = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept,
            return_inferencedata=True, chains=4, cores=4,
            nuts_sampler="numpyro",
            idata_kwargs={"log_likelihood": True}
        )

    return model, trace


# Interaction terms
predictors["ge_abs_x_gain"]  = predictors["ge_abs"] * predictors["gain"]
predictors["ge_signed_x_gain"]  = predictors["ge_signed"] * predictors["gain"]



# ============================================================
# Multi-predictor model specs M14-M27
# ============================================================

# model_specs_multi = {
#     "M14_ge_abs_gain"                     : ["ge_abs", "gain"],
#     "M15_ge_signed_gain"                : ["ge_signed", "gain"],
#     "M16_ge_abs_gain_ge_signed"           : ["ge_abs", "ge_signed", "gain"],
#     "M17_ge_abs_gain_dist"           : ["ge_abs", "gain", "dist_cursor_amp"],
#     "M18_ge_abs_gain_ge_signed_dist" : ["ge_abs", "ge_signed", "gain",
#                                              "dist_cursor_amp"],
#     "M19_gain_ge_signed_dist" : [ "ge_signed", "gain",
#                                             "dist_cursor_amp"],                                         
#     "M20_sacc_amp_gain"            : ["gain", "sacc_amp"],
#     "M21_sacc_amp_gain_dist"            : ["gain", "sacc_amp", "dist_cursor_amp"],
#     "M22_gain_dist"            : ["gain", "sacc_amp", "dist_cursor_amp"],
#     "M23_ge_abs_gain_dist_sacc_amp"   : ["ge_abs","gain", "dist_cursor_amp", 
#                                             "sacc_amp"],
#     "M24_gain_ge_signed_dist_sacc_amp"   : ["ge_signed","gain", "dist_cursor_amp", 
#                                             "sacc_amp"],
#     "M25_ge_abs_gain_ge_signed_dist_sacc_amp"   : ["ge_abs","ge_signed","gain", "dist_cursor_amp", 
#                                                 "sacc_amp"],                                                                                                                                                                                                                                                                                                                                                              
#     "M26_ge_abs_x_gain"                   : ["ge_abs", "gain",
#                                              "ge_abs_x_gain"],                                
#     "M27_ge_signed_x_gain"                  : ["ge_signed", "gain",
#                                                "ge_signed_x_gain"],
# }


# model_specs_multi = {
#     "M13_ge_abs_gain"                     : ["ge_abs", "gain"],
#     "M14_ge_signed_gain"                : ["ge_signed", "gain"],
#     "M15_ge_abs_gain_ge_signed"           : ["ge_abs", "ge_signed", "gain"],                                        
#     "M16_sacc_amp_gain"            : ["gain", "sacc_amp"],
#     "M17_sacc_amp_gain_dist"            : ["gain", "sacc_amp", "dist_cursor_amp"],
#     "M18_gain_dist"            : ["gain", "sacc_amp", "dist_cursor_amp"],
#     "M19_ge_abs_gain_dist_sacc_amp"   : ["ge_abs","gain", "dist_cursor_amp", 
#                                             "sacc_amp"],
#     "M20_gain_ge_signed_dist_sacc_amp"   : ["ge_signed","gain", "dist_cursor_amp", 
#                                             "sacc_amp"],
#     "M21_ge_abs_gain_ge_signed_dist_sacc_amp"   : ["ge_abs","ge_signed","gain", "dist_cursor_amp", 
#                                                 "sacc_amp"],                                                                                                                                                                                                                                                                                                                                                              
#     "M22_ge_abs_x_gain"                   : ["ge_abs", "gain",
#                                              "ge_abs_x_gain"],                                
#     "M23_ge_signed_x_gain"                  : ["ge_signed", "gain",
#                                                "ge_signed_x_gain"],
# }


model_specs_multi = {
    "M13_ge_abs_gain"                     : ["ge_abs", "gain"],
    "M14_ge_signed_gain"                : ["ge_signed", "gain"],
    "M15_ge_abs_gain_ge_signed"           : ["ge_abs", "ge_signed", "gain"],                                        
    "M16_sacc_amp_gain"            : ["gain", "sacc_amp"],
    "M17_sacc_amp_gain_dist"            : ["gain", "sacc_amp", "dist_cursor_amp"],
    "M18_gain_dist"                     : ["gain", "sacc_amp", "dist_cursor_amp"],                                                                                                                                                                                                                                                                                                                                                              
    "M19_ge_abs_x_gain"                   : ["ge_abs", "gain",
                                             "ge_abs_x_gain"],                                
    "M20_ge_signed_x_gain"                  : ["ge_signed", "gain",
                                               "ge_signed_x_gain"],
    "M21_ge_signed_x_gain_ge_abs"              : ["ge_signed", "gain",
                                            "ge_signed_x_gain", "ge_abs"], 
    "M22_ge_abs_x_gain_ge_signed"                     : ["ge_signed", "gain",
                                                 "ge_abs", "ge_abs_x_gain"],  
    "M23_ge_signed_x_gain_ge_abs_x_gain"      : ["ge_signed", "gain",
                                            "ge_signed_x_gain", "ge_abs", "ge_abs_x_gain"],                                                                                   
}

# ============================================================
# Run all models M14-M27
# ============================================================

traces_multi       = {}
summaries_multi    = {}
loo_results_multi  = {}
waic_results_multi = {}
failed_multi       = []

for model_name, pred_list in model_specs_multi.items():
    print(f"\n{'='*60}")
    print(f"Fitting: {model_name}")
    print(f"Predictors: {pred_list}")

    try:
        _, trace = build_and_sample_model(model_name, pred_list)
        traces_multi[model_name] = trace

        az.to_netcdf(trace, f"summary_stats/{model_name}.nc")

        var_names = ["mu_a"] + [f"mu_b_{p}" for p in pred_list] + \
                    ["sigma_a"] + [f"sigma_b_{p}" for p in pred_list]
        summary = az.summary(trace, var_names=var_names, hdi_prob=0.94)
        summaries_multi[model_name] = summary
        summary.to_excel(f"summary_stats/{model_name}_summary.xlsx", index=True)

        print(f"\nSummary — {model_name}:")
        print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]])

        loo = az.loo(trace, pointwise=True)
        loo_results_multi[model_name] = loo
        print(f"\nLOO — {model_name}: elpd = {loo.elpd_loo:.2f} ± {loo.se:.2f}")

        waic = az.waic(trace)
        waic_results_multi[model_name] = waic
        print(f"WAIC — {model_name}: elpd = {waic.elpd_waic:.2f}")

    except Exception as e:
        print(f"  FAILED: {model_name} — {e}")
        failed_multi.append(model_name)

print(f"\nFailed models: {failed_multi}")

# ============================================================
# LOO comparison table — M14-M27
# ============================================================

rows_multi = []
for model_name, loo_res in loo_results_multi.items():
    rows_multi.append({
        "model"    : model_name,
        "elpd_loo" : loo_res.elpd_loo,
        "p_loo"    : loo_res.p_loo,
        "se"       : loo_res.se,
    })

loo_df_multi = pd.DataFrame(rows_multi).sort_values(
    "elpd_loo").reset_index(drop=True)
loo_df_multi["elpd_diff"] = loo_df_multi["elpd_loo"] - \
                             loo_df_multi["elpd_loo"].iloc[0]

print("\nMulti-predictor LOO Comparison (ranked):")
print(loo_df_multi.to_string())
loo_df_multi.to_excel(
    "summary_stats/loo_multi_predictors_comparison.xlsx", index=False)

# ============================================================
# Slope summary table — M14-M27
# ============================================================

rows_slopes = []
for model_name, trace in traces_multi.items():
    pred_list = model_specs_multi[model_name]
    for pname in pred_list:
        try:
            s = az.summary(trace, var_names=[f"mu_b_{pname}"], hdi_prob=0.94)
            for i, row in s.iterrows():
                rows_slopes.append({
                    "model"    : model_name,
                    "predictor": pname,
                    "block"    : i,
                    "mean"     : row["mean"],
                    "sd"       : row["sd"],
                    "hdi_3%"   : row["hdi_3%"],
                    "hdi_97%"  : row["hdi_97%"],
                    "r_hat"    : row["r_hat"],
                    "elpd_loo" : loo_results_multi[model_name].elpd_loo
                                 if model_name in loo_results_multi else np.nan
                })
        except Exception as e:
            print(f"Summary failed for {model_name}/{pname}: {e}")

slope_df_multi = pd.DataFrame(rows_slopes)
slope_df_multi.to_excel(
    "summary_stats/slope_summary_multi_models.xlsx", index=False)
print(slope_df_multi.to_string())

# ============================================================
# Plots
# ============================================================

plt.figure(figsize=(12, 6))
plt.barh(loo_df_multi["model"], loo_df_multi["elpd_loo"],
         xerr=loo_df_multi["se"],
         color="steelblue", alpha=0.7, capsize=4)
plt.xlabel("ELPD LOO (bigger = better)")
plt.title("LOO Comparison: Multi-Predictor Models M14–M27")
plt.tight_layout()
plt.savefig("remote_plots/loo_multi_predictors.png", dpi=150)
plt.show()

# Forest plot — interaction slopes
interaction_preds = ["ge_abs_x_gain", "ge_abs_x_mov_e", "ge_abs_x_dist"]
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

for ax, pname in zip(axes, interaction_preds):
    model_name = "M23_ge_signed_x_gain_ge_abs_x_gain"
    if model_name not in traces_multi:
        continue
    s = az.summary(traces_multi[model_name],
                   var_names=[f"mu_b_{pname}"], hdi_prob=0.94)
    means  = s["mean"].values
    lo     = s["hdi_3%"].values
    hi     = s["hdi_97%"].values
    blocks = [f"B{b}" for b in unique_blocks]

    ax.errorbar(means, np.arange(len(blocks)),
                xerr=[means - lo, hi - means],
                fmt='o', color='darkorange', capsize=4)
    ax.axvline(0, linestyle='--', color='gray', alpha=0.7)
    ax.set_yticks(np.arange(len(blocks)))
    ax.set_yticklabels(blocks)
    ax.set_title(pname, fontsize=9)
    ax.set_xlabel("interaction slope", fontsize=8)

plt.suptitle("Interaction Slopes (mu_b) per Block — 94% HDI", fontsize=10)
plt.tight_layout()
plt.savefig("remote_plots/interaction_slopes_forest.png", dpi=150)
plt.show()

# %%


# %%
# ============================================================
# Proper ArviZ LOO comparison
# ============================================================

# Compute LOO for every fitted model
loo_dict = {}

for model_name, trace in traces_multi.items():
    print(f"Computing LOO: {model_name}")
    loo_dict[model_name] = az.loo(trace, pointwise=True)

# ArviZ comparison table
cmp = az.compare(
    loo_dict,
    ic="loo",
    method="stacking",      # recommended
    scale="log"             # default for elpd
)

print("\n")
print("=" * 80)
print("ARVIZ LOO MODEL COMPARISON")
print("=" * 80)
print(cmp)

# Save
cmp.to_excel(
    "summary_stats/arviz_loo_comparison.xlsx",
    index=True
)

# Nicely formatted table
cmp_reset = cmp.reset_index().rename(
    columns={"index": "model"}
)

print("\nRanked models:")
print(
    cmp_reset[
        [
            "model",
            "rank",
            "elpd_loo",
            "p_loo",
            "elpd_diff",
            "weight",
            "se",
            "dse"
        ]
    ].to_string(index=False)
)

# ============================================================
# Plot comparison
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

az.plot_compare(
    cmp,
    insample_dev=False,
    ax=ax
)

plt.title("LOO Model Comparison (ArviZ)")
plt.tight_layout()
plt.savefig(
    "remote_plots/arviz_loo_comparison.png",
    dpi=150
)
plt.show()

# ============================================================
# Best model
# ============================================================

best_model = cmp.index[0]

print("\n" + "=" * 80)
print(f"BEST MODEL: {best_model}")
print("=" * 80)

print(cmp.loc[best_model])
