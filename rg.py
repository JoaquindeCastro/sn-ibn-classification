import equinox as eqx
import jaxopt
import numpy as np
import pandas as pd
import tinygp
import jax
import jax.numpy as jnp
from io import StringIO
import matplotlib.pyplot as plt
from jax._src.config import config
from astropy.time import Time
import os
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
import seaborn as sns
import glob
from tqdm import tqdm
from multiprocessing import Pool
from astropy.cosmology import Planck18 as cosmo
from extinction import fitzpatrick99
import multiprocessing as mp
import math
from jax.scipy.special import ndtr  # standard normal CDF for censored likelihood

# ----------------------------------
# Original user configuration kept
# ----------------------------------
flts=['r','g']
col_flts_master = {
    'r': 'tab:red',
    'g': 'tab:green'
}
col_flts = {i: color for i, color in enumerate(col_flts_master.values())}
num_bands = len(flts)
num_bands = 2

ROOT_DIR = r'C:\Users\jgmad\Research\Ibn'
DATA_DIR =  os.path.join(ROOT_DIR, "data")
PLOT_DIR =  os.path.join(ROOT_DIR, "plots")
#PHOT_DIR = os.path.join(DATA_DIR, "ztf_ibn")
folders = ['ZTFBTS','ibn_papers']
PHOT_DIRS = [os.path.join(DATA_DIR, f) for f in folders]

phot_files = []
for d in PHOT_DIRS:
    phot_files.extend(glob.glob(os.path.join(d, "*")))

PLOT = True
IBN = False
DAYS_AFTER = 5
RUN_NAME = 'SN'
UNLABELED = False
RUN_NAME = f'{RUN_NAME}{"_Ibn" if IBN else ""}{"_UNLABELED" if UNLABELED else ""}'
SNT = 3
ZP = 27.5

# Extinction stuff
R_V = 3.1
lambda_eff = {'g': 4770, 'r': 6231}

# Get label/type
summary_file, = glob.glob(os.path.join(DATA_DIR, "ZTFBTS_summary.csv"))
summary_data = pd.read_csv(summary_file)
summary_data.set_index('ZTFID', inplace=True)
summary_data.replace('-', np.nan, inplace=True)
summary_data['A_V'] = pd.to_numeric(summary_data['A_V'], errors='coerce')
summary_data['redshift'] = pd.to_numeric(summary_data['redshift'], errors='coerce')

no_type_ztfids = summary_data[summary_data['type'].isna()].index.tolist()

# ----------------------------------
# Plot config (unchanged)
# ----------------------------------

def plot_config():
    sns.set_context("poster")
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style("ticks", {"xtick.major.size": 15, "ytick.major.size": 20})
    sns.set_style("ticks", {"xtick.minor.size": 8, "ytick.minor.size": 8})
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Serif",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": True,
        "xtick.major.size": 15,
        "ytick.major.size": 20,
        "xtick.minor.size": 8,
        "ytick.minor.size": 8,
    })

# ----------------------------------
# Helpers retained
# ----------------------------------

def pad_array(arr, pad_value=np.nan,max_len=100):
    padded = np.full((max_len,), pad_value)
    padded[:len(arr)] = arr[:max_len]
    return padded

def pad_inputs(t, y, yerr,band_idx, max_len=100):
    return (
        jnp.array(pad_array(np.array(t), pad_value=np.nan, max_len=max_len)),
        jnp.array(pad_array(np.array(y), pad_value=np.nan, max_len=max_len)),
        jnp.array(pad_array(np.array(yerr), pad_value=1e10,max_len=max_len)),
        jnp.array(pad_array(np.array(band_idx), pad_value=0,max_len=max_len)),
    )

import re

def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

# ----------------------------------
# Multiband two-timescale kernel (unchanged structure)
# ----------------------------------
class MultibandSeparateTimescales(tinygp.kernels.Kernel, eqx.Module):
    band_kernel_short: jnp.ndarray = eqx.field(init=False)
    band_kernel_long: jnp.ndarray = eqx.field(init=False)
    time_kernel_short: tinygp.kernels.Kernel
    time_kernel_long: tinygp.kernels.Kernel

    def __init__(
        self,
        time_kernel_short,
        time_kernel_long,
        diagonal_short,
        off_diagonal_short,
        diagonal_long,
        off_diagonal_long,
    ):
        ndim = diagonal_short.size
        factor_short = jnp.zeros((ndim, ndim))
        factor_short = factor_short.at[jnp.diag_indices(ndim)].add(diagonal_short)
        factor_short = factor_short.at[jnp.tril_indices(ndim, -1)].add(off_diagonal_short)
        self.__setattr__("band_kernel_short", factor_short @ factor_short.T)
        self.__setattr__("time_kernel_short", time_kernel_short)

        ndim2 = diagonal_long.size
        factor_long = jnp.zeros((ndim2, ndim2))
        factor_long = factor_long.at[jnp.diag_indices(ndim2)].add(diagonal_long)
        factor_long = factor_long.at[jnp.tril_indices(ndim2, -1)].add(off_diagonal_long)
        self.__setattr__("band_kernel_long", factor_long @ factor_long.T)
        self.__setattr__("time_kernel_long", time_kernel_long)

    def evaluate(self, X1, X2):
        t1, b1 = X1
        t2, b2 = X2
        shared_short = self.band_kernel_short[b1, b2] * self.time_kernel_short.evaluate(t1, t2)
        shared_long = self.band_kernel_long[b1, b2] * self.time_kernel_long.evaluate(t1, t2)
        indep = jnp.where(
            b1 == b2,
            self.time_kernel_short.evaluate(t1, t2) + self.time_kernel_long.evaluate(t1, t2),
            0.0,
        )
        return shared_short + shared_long + indep

# ----------------------------------
# Build GP (detections array only)
# ----------------------------------

def build_gp(params, t_det, yerr_det, band_idx_det):
    time_kernel_short = tinygp.kernels.Matern32(jnp.exp(params["log_scale_short"]))
    time_kernel_long  = tinygp.kernels.Matern32(jnp.exp(params["log_scale_long"]))

    kernel = MultibandSeparateTimescales(
        time_kernel_short=time_kernel_short,
        time_kernel_long=time_kernel_long,
        diagonal_short=jnp.exp(params["log_diagonal_short"]),
        off_diagonal_short=params["off_diagonal_short"],
        diagonal_long=jnp.exp(params["log_diagonal_long"]),
        off_diagonal_long=params["off_diagonal_long"],
    )
    diag = jnp.clip(yerr_det, 1e-6, 1.0) ** 2 + jnp.exp(2 * params["log_jitter"][band_idx_det])
    mean_fn = lambda x: params["mean"][x[1]]
    return tinygp.GaussianProcess(kernel, (t_det, band_idx_det), diag=diag, mean=mean_fn)

# ----------------------------------
# Censored likelihood (NEW): include non-detections as upper limits
# ----------------------------------

def loglik_censored(params,
                    t_det, b_det, y_det, yerr_det,
                    t_lim, b_lim, y_lim):
    B = num_bands
    # No detections → use prior predictive at limit locations
    if t_det.size == 0:
        # approximate observational term for limits
        pseudo_yerr = jnp.ones_like(t_lim) * (1.0 / max(SNT, 1))
        gp_prior = build_gp(params, t_lim, pseudo_yerr, b_lim)
        mu = jnp.array([params["mean"][bi] for bi in b_lim])
        var = gp_prior.kernel.evaluate((t_lim, b_lim), (t_lim, b_lim)).diagonal() + pseudo_yerr**2 + jnp.exp(2.0 * params["log_jitter"][b_lim])
        z = (y_lim - mu) / jnp.sqrt(jnp.clip(var, 1e-12, None))
        return jnp.sum(jnp.log(jnp.clip(ndtr(z), 1e-12, 1.0)))

    # Detections GP
    gp = build_gp(params, t_det, jnp.clip(yerr_det, 1e-6, 1.0), b_det)
    ll_det = gp.log_probability(y_det)

    # No limits
    if t_lim.size == 0:
        return ll_det

    # Posterior predictive at limit locations
    mu, var = gp.predict(y_det, X_test=(t_lim, b_lim), return_var=True)
    var_obs = var + (1.0 / max(SNT,1))**2 + jnp.exp(2.0 * params["log_jitter"][b_lim])
    z = (y_lim - mu) / jnp.sqrt(jnp.clip(var_obs, 1e-12, None))
    logcdf = jnp.log(jnp.clip(ndtr(z), 1e-12, 1.0))
    return ll_det + jnp.sum(logcdf)

@jax.jit
def loss_fn(params,
            t_det, b_det, y_det, yerr_det,
            t_lim, b_lim, y_lim):
    return -loglik_censored(params, t_det, b_det, y_det, yerr_det, t_lim, b_lim, y_lim)

@jax.jit
def fit_gp(params0,
           t_det, b_det, y_det, yerr_det,
           t_lim, b_lim, y_lim):
    solver = jaxopt.BFGS(fun=lambda p: loss_fn(p, t_det, b_det, y_det, yerr_det, t_lim, b_lim, y_lim),
                         linesearch=jaxopt.ZoomLineSearch(fun=lambda p: loss_fn(p, t_det, b_det, y_det, yerr_det, t_lim, b_lim, y_lim), max_stepsize=10.0),
                         verbose=0)
    soln = solver.run(params0)
    return soln.params

# ----------------------------------
# Dataframe builder (kept as in user code)
# ----------------------------------

def build_df(params_dict):
    flat_dict = {}
    for key, value in params_dict.items():
        arr = np.array(value, ndmin=1)
        if arr.size == 1:
            flat_dict[key] = arr.item()
        else:
            # Map first two entries to r,g when length==2; otherwise index by number
            for i, val in enumerate(arr):
                if arr.size == 2 and i < len(flts):
                    flat_dict[f"{key}_{flts[i]}"] = val
                else:
                    flat_dict[f"{key}_{i}"] = val
    return pd.DataFrame([flat_dict])

# ----------------------------------
# Main worker
# ----------------------------------

def looper(file):
    config.update("jax_enable_x64", True)
    if PLOT:
        plot_config()

    filename = os.path.basename(file)
    supernova_name = safe_filename(filename.split("_")[0]).split('.')[0]

    # Load
    data = pd.read_csv(file)

    # Normalize ANT-style columns if present
    if 'ant_passband' in data.columns:
        data['ant_passband'] = data['ant_passband'].replace('R', 'r')
        data = data.rename(columns={
            'ant_mjd':'mjd',
            'ant_passband':'filter',
            'ant_mag':'mag',
            'ant_magerr':'dmag',
            'ant_maglim':'maglim'
        })

    # Ensure numeric
    for col in ['mjd','mag','dmag','maglim']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Band mapping (keep original behavior start=1 → later subtract 1)
    band_map = {band: i for i, band in enumerate(col_flts_master.keys(), start=1)}

    # Extinction & time dilation when available
    if supernova_name in summary_data.index:
        A_V = float(summary_data.at[supernova_name, 'A_V']) if pd.notna(summary_data.at[supernova_name, 'A_V']) else np.nan
        z = float(summary_data.at[supernova_name, 'redshift']) if pd.notna(summary_data.at[supernova_name, 'redshift']) else np.nan
        if not np.isnan(A_V):
            ebv = A_V / R_V
            for f in ['g','r']:
                if 'filter' in data:
                    m = data['filter'].eq(f)
                    if m.any():
                        # Use a scalar band extinction to avoid shape mismatches between masks
                        A_band = float(fitzpatrick99(np.array([lambda_eff[f]]), ebv * R_V, r_v=R_V)[0])
                        if 'mag' in data:
                            mask_mag = m & data['mag'].notna()
                            if mask_mag.any():
                                data.loc[mask_mag, 'mag'] = data.loc[mask_mag, 'mag'] - A_band
                        if 'maglim' in data:
                            mask_lim = m & data['maglim'].notna()
                            if mask_lim.any():
                                data.loc[mask_lim, 'maglim'] = data.loc[mask_lim, 'maglim'] - A_band
        if not np.isnan(z) and z > 0 and 'mjd' in data:
            mjd0 = data['mjd'].min()
            data['mjd'] = (data['mjd'] - mjd0) / (1.0 + z) + mjd0
    else:
        print(f"Warning: {supernova_name} not in summary_data, skipping extinction/time-dilation.")

    # Keep only g,r
    if 'filter' in data:
        data = data[data['filter'].isin(['g','r'])]
    if len(data) == 0:
        return None, None

    # Detection vs limit masks
    has_mag = data['mag'].notna() if 'mag' in data else pd.Series([False]*len(data))
    has_dmag = data['dmag'].notna() if 'dmag' in data else pd.Series([False]*len(data))
    has_limit = data['maglim'].notna() if 'maglim' in data else pd.Series([False]*len(data))

    snr = (1.0857 / data['dmag']).replace([np.inf, -np.inf], np.nan) if 'dmag' in data else pd.Series([np.nan]*len(data))
    is_det = has_mag & (snr > SNT)
    is_lim = (~has_mag) & has_limit

    # Minimal windowing
    if is_det.any():
        first_det = data.loc[is_det, 'mjd'].min()
        last_det  = data.loc[is_det, 'mjd'].max()
        data = data[(data['mjd'] >= first_det - (DAYS_AFTER if DAYS_AFTER<0 else 15)) & (data['mjd'] <= last_det + (DAYS_AFTER if DAYS_AFTER>0 else 150))]
        # recompute masks post-cut
        has_mag = data['mag'].notna() if 'mag' in data else pd.Series([False]*len(data))
        has_limit = data['maglim'].notna() if 'maglim' in data else pd.Series([False]*len(data))
        snr = (1.0857 / data['dmag']).replace([np.inf, -np.inf], np.nan) if 'dmag' in data else pd.Series([np.nan]*len(data))
        is_det = has_mag & (snr > SNT)
        is_lim = (~has_mag) & has_limit

    # Optionally exclude labeled non-unlabeled
    if UNLABELED:
        if supernova_name not in no_type_ztfids:
            return None, None
    else:
        # If not unlabeled and peakt exists, optionally narrow window (kept from original)
        if supernova_name in summary_data.index and pd.notna(summary_data.at[supernova_name, 'peakt']):
            peak_mjd = summary_data.at[supernova_name, 'peakt'] + (2458000 - 2400000.5)
            delta = data['mjd'] - peak_mjd
            data = data[(delta >= -14) & (delta < 140)]
            # recompute masks
            has_mag = data['mag'].notna() if 'mag' in data else pd.Series([False]*len(data))
            has_limit = data['maglim'].notna() if 'maglim' in data else pd.Series([False]*len(data))
            snr = (1.0857 / data['dmag']).replace([np.inf, -np.inf], np.nan) if 'dmag' in data else pd.Series([np.nan]*len(data))
            is_det = has_mag & (snr > SNT)
            is_lim = (~has_mag) & has_limit

    if len(data) == 0 or (not is_det.any() and not is_lim.any()):
        return None, None

    # Build arrays for detections
    ddf = data[is_det].copy()
    if len(ddf) > 0:
        ddf['BAND'] = ddf['filter'].map(band_map).astype("Int32")
        flux = 10**((ddf['mag'] + 48.6)/-2.5) * 1e29
        fluxerr = (flux * ddf['dmag']) / 1.0857
        t_det = jnp.array(ddf['mjd'].values)
        y_det = jnp.log(np.clip(flux.values, 0.0, None) + 1.0)
        yerr_det = jnp.clip(fluxerr.values / np.clip(flux.values, 1e-12, None), 1e-6, 1.0)
        b_det = jnp.array(ddf['BAND'].values - 1, dtype=jnp.int32)
    else:
        t_det = jnp.array([])
        y_det = jnp.array([])
        yerr_det = jnp.array([])
        b_det = jnp.array([], dtype=jnp.int32)

    # Build arrays for limits (upper limits)
    ldf = data[is_lim].copy()
    if len(ldf) > 0:
        ldf['BAND'] = ldf['filter'].map(band_map).astype("Int32")
        flux_lim = 10**((ldf['maglim'] + 48.6)/-2.5) * 1e29
        t_lim = jnp.array(ldf['mjd'].values)
        y_lim = jnp.log(np.clip(flux_lim.values, 0.0, None) + 1.0)
        b_lim = jnp.array(ldf['BAND'].values - 1, dtype=jnp.int32)
    else:
        t_lim = jnp.array([])
        y_lim = jnp.array([])
        b_lim = jnp.array([], dtype=jnp.int32)

    # Guard: if still nothing usable
    if t_det.size == 0 and t_lim.size == 0:
        return None, None

    # Initial parameters
    def band_means_from(y, b):
        means = []
        global_mean = jnp.nanmedian(y) if y.size>0 else 0.0
        for i in range(num_bands):
            sel = y[b==i]
            means.append(jnp.nanmedian(sel) if sel.size>0 else global_mean)
        return jnp.array(means)

    params0 = {
        "mean": band_means_from(y_det, b_det),
        "log_scale_short": jnp.log(5.0),
        "log_scale_long": jnp.log(100.0),
        "log_diagonal_short": jnp.zeros(num_bands),
        "off_diagonal_short": jnp.zeros(((num_bands - 1) * num_bands) // 2),
        "log_diagonal_long": jnp.zeros(num_bands),
        "off_diagonal_long": jnp.zeros(((num_bands - 1) * num_bands) // 2),
        "log_jitter": jnp.log(jnp.full((num_bands,), 1e-3)),
    }

    # Optimize
    soln_params = fit_gp(params0, t_det, b_det, y_det, yerr_det, t_lim, b_lim, y_lim)

    # Build GP/posterior for prediction
    if t_det.size > 0:
        gp = build_gp(soln_params, t_det, yerr_det, b_det)
        predict = lambda X: gp.predict(y_det, X_test=X, return_var=False)
    else:
        # Prior predictive
        pseudo_yerr = jnp.ones_like(t_lim) * (1.0 / max(SNT,1))
        gp = build_gp(soln_params, t_lim, pseudo_yerr, b_lim)
        predict = lambda X: gp.predict(X_test=X, return_var=False)

    # Feature extraction: work in log-flux domain
    if t_det.size == 0:
        # fallback to all times in data
        tmin, tmax = float(data['mjd'].min()), float(data['mjd'].max())
    else:
        tmin, tmax = float(jnp.min(t_det)), float(jnp.max(t_det))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        return None, None

    t_sample = np.linspace(tmin, tmax, 200)
    t_sample_j = jnp.array(t_sample)
    t_sample_r = (t_sample_j, jnp.zeros_like(t_sample_j, dtype=jnp.int32))
    t_sample_g = (t_sample_j, jnp.ones_like(t_sample_j, dtype=jnp.int32))

    mu_r = np.array(predict(t_sample_r)) - 1.0
    mu_g = np.array(predict(t_sample_g)) - 1.0

    # Need both bands for color features
    if mu_r.size == 0 or mu_g.size == 0 or np.all(~np.isfinite(mu_r)) or np.all(~np.isfinite(mu_g)):
        return None, 0

    # Restrict to post-first-detection interval when available
    first_det_time = float(tmin)
    valid_mask = (t_sample >= first_det_time) & (t_sample <= tmax)
    t0 = t_sample[valid_mask]
    mu_r = mu_r[valid_mask]
    mu_g = mu_g[valid_mask]

    # Color in log-flux space (g - r); convert to mag difference with -1.0857 factor if needed downstream
    color = mu_g - mu_r
    if color.size == 0:
        return None, 0

    idx_peak_r = int(np.nanargmax(mu_r))
    idx_peak_g = int(np.nanargmax(mu_g))

    color_at_r_peak = float(color[idx_peak_r])
    color_at_g_peak = float(color[idx_peak_g])
    color_mean = float(np.nanmean(color))

    peak_time = t0[idx_peak_r]
    rise_mask = (t0 >= first_det_time) & (t0 <= peak_time)
    if np.count_nonzero(rise_mask) > 1:
        dt = t0[rise_mask] - t0[rise_mask].mean()
        dc = color[rise_mask] - color[rise_mask].mean()
        color_slope = float((dt*dc).sum()/(dt*dt).sum())
    else:
        color_slope = np.nan

    min_idx = int(np.nanargmin(color))
    max_idx = int(np.nanargmax(color))
    color_min_time = float(t0[min_idx] - first_det_time)
    color_max_time = float(t0[max_idx] - first_det_time)

    # Optional plotting (robust to absence of source indices)
    if PLOT:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.figure(figsize=(15,8))
        # detections
        if len(ddf) > 0:
            for n, band_code in enumerate(['r','g']):
                m = (ddf['filter']==band_code)
                if m.any():
                    t_plot = ddf.loc[m, 'mjd'].values
                    F = 10**((ddf.loc[m,'mag'].values + 48.6)/-2.5) * 1e29
                    Ferr = (F * ddf.loc[m,'dmag'].values)/1.0857
                    plt.errorbar(t_plot, F, Ferr, fmt='o', mec='k', mfc=col_flts[n], ms=8, color=col_flts[n], label=f'{band_code} det')
        # limits
        if len(ldf) > 0:
            for n, band_code in enumerate(['r','g']):
                m = (ldf['filter']==band_code)
                if m.any():
                    t_plot = ldf.loc[m, 'mjd'].values
                    Flim = 10**((ldf.loc[m,'maglim'].values + 48.6)/-2.5) * 1e29
                    plt.scatter(t_plot, Flim, marker='v', s=50, c=col_flts[n], label=f'{band_code} limit')
        # GP curves
        t_test_j = jnp.linspace(float(data['mjd'].min())-50, float(data['mjd'].max())+50, 600)
        for n, bi in enumerate([0,1]):
            bi_arr = jnp.full_like(t_test_j, bi, dtype=jnp.int32)
            mu = np.array(predict((t_test_j, bi_arr)))
            plt.plot(np.array(t_test_j), np.exp(mu)-1.0, color=col_flts[n])
        plt.xlabel("Time [MJD]")
        plt.ylabel("Flux [arb]")
        plt.title(f"{supernova_name} – GP Fit (censored)")
        plt.legend(loc='best')
        plot_path = os.path.join(PLOT_DIR, f"gp_fit_{RUN_NAME}_{supernova_name}.png")
        plt.tight_layout(); plt.savefig(plot_path, dpi=150)
        plt.close()

    # Save parameters
    safe_params = {}
    for k, v in soln_params.items():
        val = np.asarray(v)
        safe_params[k] = val.tolist() if val.ndim > 0 else val.item()
    Ibn_params = build_df(safe_params)
    Ibn_params["supernova_name"] = supernova_name
    Ibn_params['color_mean'] = color_mean
    Ibn_params['color_at_peak'] = color_at_r_peak
    Ibn_params["color_slope"] = color_slope
    Ibn_params["color_min_time"] = color_min_time
    Ibn_params["color_max_time"] = color_max_time
    return Ibn_params, 0

# ----------------------------------
# Misc helpers (unchanged)
# ----------------------------------

def get_sn_name_from_path(f):
    return safe_filename(os.path.basename(f).split("_")[0]).split('.')[0]

# ----------------------------------
# Main (kept structure & paths)
# ----------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    all_params = []
    plt.figure(figsize=(15, 8))

    if IBN:
        Ibns = list(summary_data[summary_data['type'] == 'SN Ibn'].index)
        files = [f for f in phot_files if safe_filename(os.path.basename(f).split("_")[0]).split('.')[0] in Ibns]
        n_parallel = 2
    else:
        files = [f for f in phot_files]
        n_parallel = 30

    save_path = f"{DATA_DIR}/gp_params_{RUN_NAME}_all{'_' + str(DAYS_AFTER) if DAYS_AFTER else ''}.csv"

    if os.path.exists(save_path):
        try:
            processed_df = pd.read_csv(save_path, usecols=["supernova_name"])
            processed_names = set(processed_df["supernova_name"].unique())
            files_to_process = [f for f in files if get_sn_name_from_path(f) not in processed_names]
        except Exception:
            files_to_process = files
    else:
        files_to_process = files

    BATCH_SIZE = max(1, math.ceil(len(files_to_process)/20))

    with Pool(processes=min(1, max(1, n_parallel)), maxtasksperchild=1) as pool:
        with tqdm(total=len(files_to_process)) as pbar:
            results = []
            batch_counter = 0
            header_written = os.path.exists(save_path)

            for params_result, l in pool.imap(looper, files_to_process):
                if params_result is not None:
                    results.append(params_result)

                if len(results) >= BATCH_SIZE:
                    df_batch = pd.concat(results, ignore_index=True)
                    df_batch.to_csv(save_path, mode='a', header=not header_written, index=False)
                    header_written = True
                    results.clear()
                    batch_counter += 1
                    print(f"[INFO] Batch {batch_counter} saved.")

                pbar.update()

            if results:
                df_batch = pd.concat(results, ignore_index=True)
                df_batch.to_csv(save_path, mode='a', header=not header_written, index=False)
                print(f"[INFO] Final batch saved.")
