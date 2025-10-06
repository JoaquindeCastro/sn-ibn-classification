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

flts=['r','g']
col_flts_master = {
    'r': 'tab:red',
    'g': 'tab:green'
}
col_flts = {i: color for i, color in enumerate(col_flts_master.values())}
num_bands = len(flts)
num_bands = 2

def pad_array(arr, pad_value=np.nan,max_len=100):
    padded = np.full((max_len,), pad_value)
    padded[:len(arr)] = arr[:max_len]  # Truncate if too long
    return padded

def pad_inputs(t, y, yerr,band_idx, max_len=100):
    return (
        jnp.array(pad_array(np.array(t), pad_value=np.nan, max_len=max_len)),
        jnp.array(pad_array(np.array(y), pad_value=np.nan, max_len=max_len)),
        jnp.array(pad_array(np.array(yerr), pad_value=1e10,max_len=max_len)),
        jnp.array(pad_array(np.array(band_idx), pad_value=0,max_len=max_len)),
    )

def build_df(params_dict):
    flat_dict = {}
    for key, value in params_dict.items():
        # Convert value to a NumPy array (just in case it's a scalar)
        arr = np.array(value, ndmin=1)
        if arr.size == 1:
            # If there's only one entry, store it directly
            flat_dict[key] = arr.item()
        else:
            # Otherwise, store each element in a separate key
            for i, val in enumerate(arr):
                flat_dict[f"{key}_{flts[i]}"] = val
                '''try:
                    flat_dict[f"{key}_{flts[i]}"] = val
                except:
                    flat_dict[key] = arr.tolist()'''

    # Build a one-row DataFrame
    df = pd.DataFrame([flat_dict])
    return df

ROOT_DIR = r'C:\Users\jgmad\Research\Ibn'
DATA_DIR =  os.path.join(ROOT_DIR, "data")
PLOT_DIR =  os.path.join(ROOT_DIR, "plots")
#PHOT_DIR = os.path.join(DATA_DIR, "ztf_ibn")
PHOT_DIR = os.path.join(DATA_DIR, "sn_all")

PLOT = True
RUN_NAME = 'SN_rg_test_pad'
SNT = 3
ZP = 27.5

# Get label/type
summary_file, = glob.glob(os.path.join(DATA_DIR, "ztf_all_summary.csv"))
summary_data = pd.read_csv(summary_file)
summary_data.set_index('ZTFID', inplace=True)

def plot_config():
    sns.set_context("poster")
    sns.set(font_scale=1.8)
    sns.set_style("white")
    sns.set_style("ticks", {"xtick.major.size": 15, "ytick.major.size": 20})
    sns.set_style("ticks", {"xtick.minor.size": 8, "ytick.minor.size": 8})
    plt.rcParams.update({
    "text.usetex": False, # Changed to false because of error
    # "font.family": "serif",
    "font.family": "DejaVu Serif",
    #"font.serif": ["serif"], # Used to be Palantino
    "xtick.minor.visible": True,     # Make minor ticks visible on the x-axis
    "ytick.minor.visible": True,     # Make minor ticks visible on the y-axis
    "xtick.direction": "in",         # Minor ticks direction for x-axis
    "ytick.direction": "in",         # Minor ticks direction for y-axis
    "xtick.top": False,               # Show ticks on the top x-axis
    "ytick.right": True,             # Show ticks on the right y-axis
    "xtick.major.size": 15,          # Major tick size for x-axis
    "ytick.major.size": 20,          # Major tick size for y-axis
    "xtick.minor.size": 8,           # Minor tick size for x-axis
    "ytick.minor.size": 8,           # Minor tick size for y-axis
    })


# (1) Rewrite build_gp so that it accepts event-specific arrays.
def build_gp(params, t, yerr, band_idx,mask):
    t = jnp.where(mask,t,0.0)
    yerr = jnp.where(mask,yerr,1e5)
    time_kernel_short = tinygp.kernels.Matern32(jnp.exp(params["log_scale_short"]))
    time_kernel_long = tinygp.kernels.Matern32(jnp.exp(params["log_scale_long"]))

    # Kernel for separate timescales; note that this uses your custom kernel.
    kernel = MultibandSeparateTimescales(
        time_kernel_short=time_kernel_short,
        time_kernel_long=time_kernel_long,
        diagonal_short=jnp.exp(params["log_diagonal_short"]),
        off_diagonal_short=params["off_diagonal_short"],
        diagonal_long=jnp.exp(params["log_diagonal_long"]),
        off_diagonal_long=params["off_diagonal_long"],
    )
    # Diagonal noise term: we index log_jitter with the band index.
    diag = yerr ** 2 + jnp.exp(2 * params["log_jitter"][band_idx])
    # The mean function is defined band-wise.
    # return NaNIgnoringGP(tinygp.GaussianProcess(kernel, (t, band_idx), diag=diag, mean=lambda x: params["mean"][x[1]]))
    return tinygp.GaussianProcess(kernel, (t, band_idx), diag=diag, mean=lambda x: params["mean"][x[1]])



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
    '''def evaluate(self, X1, X2):
        t1, b1 = X1
        t2, b2 = X2
        shared_short = self.band_kernel_short[b1, b2] * self.time_kernel_short.evaluate(t1, t2)
        shared_long = self.band_kernel_long[b1, b2] * self.time_kernel_long.evaluate(t1, t2)
        indep = jnp.where(
            b1 == b2,
            self.time_kernel_long.evaluate(t1, t2),
            0.0,
        )
        return shared_long + indep'''


import re

def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

class NaNIgnoringGP:
    def __init__(self, gp: tinygp.GaussianProcess):
        self.gp = gp

    def log_probability(self, y):
        mask = ~jnp.isnan(y)
        return self.gp.condition(y[mask], mask=mask).log_probability(y[mask])

    def predict(self, *args, **kwargs):
        return self.gp.predict(*args, **kwargs)

# (2) Wrap the GP fit (loss and optimization) in a jitted function using a JAX-native optimizer.
@jax.jit
def fit_gp(params, tp, yp, yerrp, band_idx,mask):
    def loss_fn(p):
        gp = build_gp(p, tp, yerrp, band_idx,mask)
        return -gp.log_probability(jnp.where(jnp.isnan(yp),0.0,yp))
    linesearch = jaxopt.ZoomLineSearch(fun=loss_fn, max_stepsize=10.0)
    solver = jaxopt.BFGS(fun=loss_fn, linesearch=linesearch, verbose=0)
    soln = solver.run(params)
    #print(soln.params)
    return soln.params

def looper(file):
    config.update("jax_enable_x64", True)
    plot_config()
    today_mjd = Time.now().mjd
    filename = os.path.basename(file)
    supernova_name = filename.split("_")[0]
    supernova_name = safe_filename(supernova_name)
    supernova_name = supernova_name.split('.')[0]
    data = pd.read_csv(file)
    data = data[data['magtype'] == 1] #remove upper limits from fit for now

    data.rename(columns={'mjd':'MJD'}, inplace=True)

    # Get SN type
    sn_type = summary_data.at[supernova_name, 'type'] # supposed to be faster

    # FILTER BY DATE
    peak_mjd = summary_data.at[supernova_name, 'peakt'] + (2458000 - 2400000.5)
    # Compute time difference between obs_date and disc_date
    delta = data['MJD'] - peak_mjd
    # Keep only rows within - two weeks to + 4 months

    data = data[(delta >= -14) & (delta < 140)]

    band_map = {band: i for i, band in enumerate(col_flts_master.keys(),start=1)}

    data['BAND'] = data['filter'].map(band_map).astype("Int32") # supposed to be faster than above code


    data = data.dropna(subset=["BAND"])  # Cus only doing r and g here
    # convert from apparent magnitude to flux
    data['FLUXCAL'] = 10**((data["mag"] + 48.6)/-2.5) * 1e29
    data['FLUXCALERR'] = (data['FLUXCAL'] * data["dmag"]) / 1.0857

    # FILTER SNT
    data = data.dropna(subset=['FLUXCAL','FLUXCALERR']) # remove nan values
    data = data[data['FLUXCALERR'] > 0]
    data = data[data['FLUXCAL']/data['FLUXCALERR'] > 4] # Filter out small Signal to Noise Ratio

    if len(data)==0:
        return None,None

    t = jnp.array(data["MJD"].values)
    y = jnp.log(jnp.array(data["FLUXCAL"].values) + 1)
    yerr = jnp.array(data["FLUXCALERR"].values) / jnp.array(data["FLUXCAL"].values)
    band_idx = jnp.array(data["BAND"].values - 1, dtype=jnp.int32)

    tp,yp,yerrp,band_idxp = pad_inputs(t,y,yerr,band_idx)

    ogmask = ~jnp.isnan(tp)


    source_dict = {
        'ztf':0,
        'ysepz':1,
        'papers':2
    }
    source_idx = jnp.array([source_dict[source] for source in data["source"].values], dtype=jnp.int32)
    source_idx = pad_array(source_idx)

    # define the initial parameters for the run
    params = {
        "mean": jnp.array([jnp.nanmean(y[band_idx == i]) for i in range(num_bands)]),
        "log_scale_short": jnp.log(5.0),
        "log_scale_long": jnp.log(100.0),
        "log_diagonal_short": jnp.zeros(num_bands),
        "off_diagonal_short": jnp.zeros(((num_bands - 1) * num_bands) // 2),
        "log_diagonal_long": jnp.zeros(num_bands),
        "off_diagonal_long": jnp.zeros(((num_bands - 1) * num_bands) // 2),
        "log_jitter": jnp.zeros(num_bands),
    }

    # Run the GP fit.
    soln_params = fit_gp(params, tp, yp, yerrp, band_idxp,ogmask)
    print(soln_params)

    # Build the GP with the optimized parameters for fitting.
    gp = build_gp(soln_params, tp, yerrp, band_idxp,ogmask)
    #plt.axvline(x=peak_mjd,color='black',linestyle='--')

    l = gp.log_probability(yp)

    # Plotting the results against observed data
    ztf_mask = (source_idx == 0)
    ysepz_mask = (source_idx == 1)
    paper_mask = (source_idx == 2)

    gp_upper_bounds = []

    if PLOT:

        for n in range(num_bands):
            mask = (band_idxp == n)

            t_plot = np.array(tp[mask])

            # do not plot if less than 3 datapoints
            if len(t_plot) < 3:
                
                continue
            y_plot = np.exp(np.array(yp[mask])) - 1
            # basic propagation of uncertainty
            yerr_plot = np.exp(np.array(yp[mask])) * np.array(yerrp[mask])
            mask1 = jnp.logical_and(mask, ztf_mask)
            mask2 = jnp.logical_and(mask, ysepz_mask)
            mask3 = jnp.logical_and(mask, paper_mask)

            plt.errorbar(np.array(tp[mask & ztf_mask]), np.exp(np.array(yp[mask & ztf_mask])) - 1, np.exp(np.array(yp[mask & ztf_mask])) * np.array(yerrp[mask & ztf_mask]), fmt="o", mec='k', mfc=col_flts[n], ms=10, color=col_flts[n])
            plt.errorbar(np.array(tp[mask2]), np.exp(np.array(yp[mask2])) - 1, np.exp(np.array(yp[mask2])) * np.array(yerrp[mask2]), fmt="^", mec='k', mfc=col_flts[n], ms=10, color=col_flts[n])
            plt.errorbar(np.array(tp[mask3]), np.exp(np.array(yp[mask3])) - 1, np.exp(np.array(yp[mask3])) * np.array(yerrp[mask3]), fmt="s", mec='k', mfc=col_flts[n], ms=10, color=col_flts[n])

            t_test = np.linspace(np.array(t).min()-50, np.array(t).max()+50, 1000)
            X_test = (t_test, np.full(t_test.shape, n, dtype=np.int32))

            #mean and variance of the GP fits
            print(len(np.array(y)),X_test,t_test.shape)

            mu, var = gp.predict(np.array(jnp.array(pad_array(np.array(y), pad_value=0.0)),), X_test=X_test, return_var=True)
            

            std = np.sqrt(var)
            if yerr_plot.size > 0:
                gp_upper_bounds.append(yerr_plot.max())

            plt.plot(t_test, np.exp(mu) - 1, color=col_flts[n])
            plt.fill_between(t_test, np.exp(mu - std) - 1, np.exp(mu + std) - 1, color=col_flts[n], alpha=0.3)

        #create plotting directory
        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        '''# Compute upper bound from observed data
        observed_upper = (np.exp(yp) - 1) + (np.exp(yp) * yerrp)

        # Compute upper bound from GP prediction
        # Combine all maxima
        try:
            max_val = max(observed_upper.max(), max(gp_upper_bounds))
            plt.ylim(0, max_val * 1.05)  # 5% headroom
        except ValueError:
            pass'''

        # plt.xlim(np.array(jnp.where(jnp.isnan(tp),tp,0.0)).min(), np.array(jnp.where(jnp.isnan(tp),tp,0.0)).max())
        plt.xlabel("Time [MJD]")
        plt.ylabel("Flux [DNU]")
        plt.title(f"{supernova_name} ({sn_type}), GP Fit")
        plot_path = os.path.join(PLOT_DIR, f"gp_fit_{RUN_NAME}_{supernova_name}.png")
        # plot_path = f"{PLOT_DIR}/gp_fit_{RUN_NAME}_{supernova_name}.png" Was getting error possibly due to Mac/Windows
        plt.savefig(plot_path)
        #print(f"Saved plot for {supernova_name} at {plot_path}.")

        #clear figure to re-use it
        # plt.clf()
        plt.show()

    #save parameters from the fit
    #Ibn_params = build_df(soln_params)
    safe_params = {}
    for k, v in soln_params.items():
        val = np.asarray(v)
        safe_params[k] = val.tolist() if val.ndim > 0 else val.item()
    # Ibn_params = build_df(safe_params)

    Ibn_params = build_df(safe_params)

    Ibn_params["supernova_name"] = supernova_name 
    #print(Ibn_params)
    return Ibn_params,l

if __name__ == "__main__":

    all_params = []

    phot_files = glob.glob(os.path.join(PHOT_DIR, "*"))

    plt.figure(figsize=(15, 8))
    
    files = [f for f in phot_files] # [:]
    Ibns = list(summary_data[summary_data['type'] == 'SN Ibn'].index)
    # files = [f for f in phot_files if safe_filename(os.path.basename(f).split("_")[0]).split('.')[0] in Ibns]
    # files = [f for f in phot_files if safe_filename(os.path.basename(f).split("_")[0]).split('.')[0] in ['ZTF19aayrosj']]
    print(len(files))
    
    BATCH_SIZE = 100
    save_path = f"{DATA_DIR}/gp_params_{RUN_NAME}_all.csv"

    '''results = []
    ls1 = []
    ls2 = []
    for f in files:
        r,l = looper(f)
        results.append(r)
        ls1.append(l)
    df_batch = pd.concat(results, ignore_index=True)
    df_batch.to_csv(save_path, mode='a', index=False)'''

    files = [a for a in files if 'ZTF' not in a]
    

    with Pool(processes=1, maxtasksperchild=15) as pool:
        with tqdm(total=len(files)) as pbar:
            results = []
            batch_counter = 0
            header_written = os.path.exists(save_path) 

            for params_result,l in pool.imap(looper, files):
                if params_result is not None:
                    results.append(params_result)
                    # ls2.append(l)

                if len(results) >= BATCH_SIZE:
                    # Concatenate and write batch to CSV
                    df_batch = pd.concat(results, ignore_index=True)
                    df_batch.to_csv(save_path, mode='a', header=not header_written, index=False)
                    header_written = True
                    results.clear()  # clear for next batch
                    batch_counter += 1
                    print(f"[INFO] Batch {batch_counter} saved.")

                pbar.update()

            # Save any remaining results after loop ends
            if results:
                df_batch = pd.concat(results, ignore_index=True)
                df_batch.to_csv(save_path, mode='a', header=not header_written, index=False)
                print(f"[INFO] Final batch saved.")

    '''print(ls1)
    for i in range(len(ls1)):
        print(ls1[i]-ls2[i])'''