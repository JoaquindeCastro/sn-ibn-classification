from astropy.time import Time
from astropy.io import ascii
import numpy as np
import os
from matplotlib import pyplot as plt
from extinction import fitzpatrick99
from scipy.interpolate import UnivariateSpline
from lmfit import Model
import pandas as pd
from lc_param_GP import (
    check_candidates, check_candidates_color, get_LC,
)

import glob
from tqdm import tqdm
import re
from astropy.cosmology import Planck18 as cosmo

from alerce.core import Alerce
client = Alerce()

from concurrent.futures import ProcessPoolExecutor


# added this recently
EXTRAPOLATE_RISE_TIME_IF_NO_DECLINE = True
RISE_SLOPE_FIXED_WINDOW_DAYS = 10 
MAX_EXTRAP_DAYS_AFTER_LAST = 30 

def _pairwise_color_offset(epoch_g, mag_g, epoch_r, mag_r, dt=0.5):
    """
    Robust median (g - r) using near-simultaneous pairs (|Δt| <= dt).
    Returns (offset, n_pairs). We SUBTRACT offset from g to map onto r.
    """
    eg = np.asarray(epoch_g, float); mg = np.asarray(mag_g, float)
    er = np.asarray(epoch_r, float); mr = np.asarray(mag_r, float)
    diffs = []
    if eg.size == 0 or er.size == 0:
        return 0.0, 0
    for tg, vg in zip(eg, mg):
        j = np.where(np.abs(er - tg) <= dt)[0]
        if j.size > 0:
            diffs.extend(list(vg - mr[j]))
    if len(diffs) >= 2:
        return float(np.median(diffs)), len(diffs)
    # fallback: median difference if we didn’t get enough pairs
    try:
        return float(np.median(mg) - np.median(mr)), 0
    except Exception:
        return 0.0, 0


def _combine_rg_as_white(ddf_g, ddf_r, ndf_g, ndf_r, ng, nr, dt_pair=0.5):
    """
    Build a single 'white' series from r + (g shifted by median g-r).
    Returns:
      x_all, y_all, e_all, non_epoch_rg, non_mag_rg, last_nondet_union, gr_off, gr_npairs
    """
    # Arrays (detections)
    xg = ddf_g['mjd'].values if len(ddf_g) else np.array([])
    yg = ddf_g['mag'].values if len(ddf_g) else np.array([])
    eg = ddf_g['dmag'].values if len(ddf_g) else np.array([])

    xr = ddf_r['mjd'].values if len(ddf_r) else np.array([])
    yr = ddf_r['mag'].values if len(ddf_r) else np.array([])
    er = ddf_r['dmag'].values if len(ddf_r) else np.array([])

    # color offset (g - r)
    gr_off, gr_npairs = _pairwise_color_offset(xg, yg, xr, yr, dt=dt_pair)

    # shift g onto r system
    yg_shift = yg - gr_off

    # stack detections
    x_all = np.concatenate([xr, xg])
    y_all = np.concatenate([yr, yg_shift])
    e_all = np.concatenate([er, eg])

    # sort by time
    order = np.argsort(x_all)
    x_all = x_all[order]; y_all = y_all[order]; e_all = e_all[order]

    # nondetections (shift g limits the same way)
    non_epoch_r = ndf_r['mjd'].values if len(ndf_r) else np.array([])
    non_epoch_g = ndf_g['mjd'].values if len(ndf_g) else np.array([])

    nr = np.asarray(nr, float) if len(nr) else np.array([])
    ng = np.asarray(ng, float) if len(ng) else np.array([])
    ng_shift = ng - gr_off if ng.size else np.array([])

    non_epoch_rg = np.concatenate([non_epoch_r, non_epoch_g])
    non_mag_rg   = np.concatenate([nr, ng_shift])

    # last nondet strictly before first detection (across both bands)
    last_nd = None
    if x_all.size:
        first_det = float(x_all[0])
        cands = []
        if non_epoch_r.size:
            r_prev = non_epoch_r[non_epoch_r < first_det]
            if r_prev.size: cands.append(np.max(r_prev))
        if non_epoch_g.size:
            g_prev = non_epoch_g[non_epoch_g < first_det]
            if g_prev.size: cands.append(np.max(g_prev))
        if len(cands):
            last_nd = float(np.max(cands))

    return x_all, y_all, e_all, non_epoch_rg, non_mag_rg, last_nd, float(gr_off), int(gr_npairs)


step = 1

def log(message):
    global step
    print(f'Step {step}: {message}')
    step+=1

def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

'''
Expects data to be in the following format
dateobs,mjd,mag,dmag,source,filter,magtype
'''


ROOT_DIR = r'C:\Users\jgmad\Research\Ibn'
DATA_DIR =  os.path.join(ROOT_DIR, "data")
PLOT_DIR =  os.path.join(ROOT_DIR, "plots")
#PHOT_DIR = os.path.join(DATA_DIR, "ztf_ibn")
folders = ['ZTFBTS','ibn_papers']
PHOT_DIRS = [os.path.join(DATA_DIR, f) for f in folders]

phot_files = []
for d in PHOT_DIRS:
    phot_files.extend(glob.glob(os.path.join(d, "*")))

test = ['ZTF18aaaonon','ZTF18aaaooqj']

PLOT = True
SNT = 4
ZP = 27.5
DAYS_AFTER = 0
DAYS_FROM_PEAK = 0 # only used if DAYS_AFTER=0
IBN = False
UNLABELED = False

# Get label/type
summary_file, = glob.glob(os.path.join(DATA_DIR, "ZTFBTS_summary.csv"))
summary_data = pd.read_csv(summary_file)
summary_data.set_index('ZTFID', inplace=True)

summary_data.replace('-', np.nan, inplace=True)
summary_data['A_V'] = pd.to_numeric(summary_data['A_V'], errors='coerce')
summary_data['redshift'] = pd.to_numeric(summary_data['redshift'], errors='coerce')

#type_dict = dict(zip(summary_data['ZTFID'], summary_data['type']))
no_type_ztfids = summary_data[summary_data['type'].isna()].index.tolist()

Ibns = list(summary_data[summary_data['type'] == 'SN Ibn'].index)

phot_files = [f for f in phot_files if any(x in f for x in Ibns)]

RUN_NAME = f'SN{"_Ibn" if IBN else ""}{"_UNLABELED" if UNLABELED else ""}'
today_mjd = Time.now().mjd

log('Collecting files...')

data_list_r = {}
data_list_g = {}
data_list_rg = {}  # NEW: joint r+g features

log('Fitting g and r bands...')

new_ibns = ['2011hw', '2014av', '2015U', '2018jmt', '2019deh', '2019kbj', '2019uo', '2019wep', '2020nxt', '2021jpk', '2023emq', '2023tsz', '2024acyl', 'iPTF14aki', 'iPTF15akq', 'iPTF15ul', 'ps1-12sk', 'PTF11rfh', 'PTF12ldy']
Ibns.extend(new_ibns)

# Extinction stuff
R_V = 3.1
lambda_eff = {'g': 4770, 'r': 6231}

nodets = []

def process(file):
    filename = os.path.basename(file)
    supernova_name = filename.split("_")[0]
    supernova_name = safe_filename(supernova_name)
    supernova_name = supernova_name.split('.')[0]
    if IBN: 
        if supernova_name not in Ibns:
            return None,None,None,None
    if UNLABELED:
        if supernova_name not in no_type_ztfids:
            return None, None,None,None
    
    '''if supernova_name not in ['ZTF20abfadah','2019wep','ZTF20abyznqs','ZTF21aauvmck','ZTF21achujxq','ZTF19aatmkll','ZTF19abhcefa','ZTF22aaaepgu','ZTF22aawxlpc','ZTF20aalrqbu','ZTF24abiesnr','ZTF22aahftli']:
        return None,None,None,None'''

    data = pd.read_csv(file)

    '''# convert from flux to mag
    try:
        flux_mask = data['FLUXCAL'] > 0
        data.loc[flux_mask & data['mag'].isna(), 'mag'] = (np.log10(data.loc[flux_mask, 'FLUXCAL']/1e29)*-2.5) - 48.6
        data.loc[flux_mask & data['dmag'].isna(), 'dmag'] = (data.loc[flux_mask, 'FLUXCALERR'] * 1.0857 ) / data.loc[flux_mask, 'FLUXCAL']
    except Exception as e:
        pass'''

    
    # FILTERING
    try:

        data['ant_passband'] = data['ant_passband'].replace('R', 'r') # r is labeled as R for some reason

        data = data.rename(columns={
            'ant_mjd':'mjd',
            'ant_passband':'filter',
            'ant_mag':'mag',
            'ant_magerr':'dmag'
        })

        mask_non_detection = data['mag'].isna() & data['ant_maglim'].notna()
        data['magtype'] = np.where(mask_non_detection, -1, 1)
        data = data[~((data['magtype'] == 1) & (1.086/data['dmag'] <= SNT))]
        #data['magtype'] = np.where(1.086 / data['dmag'] > 4, 1, -1)

    except Exception as e:
        print(f'now processing {supernova_name} because of {e}')

    data['mjd'] = pd.to_numeric(data['mjd'], errors='coerce')

    if supernova_name in summary_data.index:
        A_V = float(summary_data.at[supernova_name, 'A_V'])
        ebv = float(A_V / R_V)

        for f in ['g', 'r']:
            filt_mask = data['filter'] == f
            if filt_mask.any():
                A_lambda = fitzpatrick99(np.full(filt_mask.sum(), lambda_eff[f]), ebv * R_V, r_v=R_V)
                data.loc[filt_mask, 'mag'] -= A_lambda

        z = float(summary_data.at[supernova_name, 'redshift'])
        if pd.notna(z) and z != 0:
            data['mjd'] = (data['mjd'] - data['mjd'].min()) / (1. + z) + data['mjd'].min()

        '''dL = cosmo.luminosity_distance(z).to("pc").value
        mu = 5 * np.log10(dL) - 5
        print('before')
        print(data['mag'])               
        data['mag'] = data['mag'] -  mu
        print('after')
        print(data['mag'])'''
        # above does not work very sad, just including it in classifier

    else:
        print(f"Warning: {supernova_name} not in summary_data, skipping extinction and redshift correction.")

    data = data.dropna(subset=['mjd']) # remove nan values
    data = data[~( (data['magtype']==1) & (data['mag'].isna() | data['dmag'].isna()) )]
    data = data[~((data['magtype'] == 1) & (data['dmag'] <= 0))]
    #print(data[~(data['magtype'] == 1)])


    if len(data) <= 3:
        return None,None,None,None
    if len(data[(data['magtype'] == 1) & (data['filter'] == 'r')]) < 4 or len(data[(data['magtype'] == 1) & (data['filter'] == 'g')]) < 4:
        return None,None,None,None
        

    if DAYS_AFTER:
        if DAYS_AFTER>0:

            dets = data[data['magtype'] == 1]
            first_det = dets['mjd'].min()

            # Compute time difference between obs_date and first_det
            det_delta = data['mjd'] - first_det

            # Keep only rows within DAYS_AFTER after first_det
            data = data[det_delta < DAYS_AFTER]
        elif DAYS_AFTER<0:
            if supernova_name in summary_data.index:
                # need to convert peakt from ZTF to MJD
                # cut time series data to only included peakt +/- -DAYS_AFTER 
                # FILTER BY DATE
                peak_mjd = summary_data.at[supernova_name, 'peakt'] + (2458000 - 2400000.5)
                # Compute time difference between obs_date and disc_date
                delta = data['mjd'] - peak_mjd
                # Keep only rows within - two weeks to + 4 months

                data = data[(delta >= DAYS_AFTER) & (delta < -DAYS_AFTER)]
                #print(peak_mjd)
    elif isinstance(DAYS_AFTER, (int,float)) and DAYS_AFTER==0:
        if supernova_name in summary_data.index:
            # need to convert peakt from ZTF to MJD
            # cut time series data to only included peakt +/- -DAYS_AFTER 
            # FILTER BY DATE
            peak_mjd = summary_data.at[supernova_name, 'peakt'] + (2458000 - 2400000.5)
            data['days_from_peak'] = data['mjd'] - peak_mjd
            if DAYS_FROM_PEAK is not None:
                data = data[data['days_from_peak'] < DAYS_FROM_PEAK]


    ddf = data[(data['magtype'] == 1)]
    ndf = data[~(data['magtype'] == 1) & (data['mjd'] < ddf['mjd'].min())]
    ddf_r = ddf[ddf['filter'] == 'r']
    ndf_r = ndf[ndf['filter'] == 'r']
    ddf_g = ddf[ddf['filter'] == 'g']
    ndf_g = ndf[ndf['filter'] == 'g']
    if 'ant_maglim' in data:
        nr = ndf_r['ant_maglim'].values
        ng = ndf_g['ant_maglim'].values
    else:
        nr = []
        ng = []

    '''
    For plotting Ibns
    '''

    data_list_r_row = check_candidates(oid=f'{supernova_name} (r)', filt='r', ifplot=PLOT,ifselfdata=True,
                    epoch=ddf_r['mjd'].values,
                    mag=ddf_r['mag'].values,
                    mag_err=ddf_r['dmag'].values,
                    non_epoch=ndf_r['mjd'].values,
                    non_mag=nr,
                    last_nondet=[ndf_r["mjd"].max()] if not ndf.empty else []
    )

    data_list_g_row = check_candidates(oid=f'{supernova_name} (g)' , filt='g', ifplot=PLOT,ifselfdata=True,
                    epoch=ddf_g['mjd'].values,
                    mag=ddf_g['mag'].values,
                    mag_err=ddf_g['dmag'].values,
                    non_epoch=ndf_g['mjd'].values,
                    non_mag=ng,
                    last_nondet=[ndf_g["mjd"].max()] if not ndf.empty else []
    )

    # --- NEW: combine r and g here, then call the same check_candidates() ---
    data_list_rg_row = None
    try:
        x_all, y_all, e_all, non_epoch_rg, non_mag_rg, last_nd_union, gr_off, gr_npairs = _combine_rg_as_white(
            ddf_g, ddf_r, ndf_g, ndf_r, ng, nr, dt_pair=0.5
        )
        if min(len(ddf_r['mag'].values), len(ddf_g['mag'].values)) > x_all.size:
            # sanity check: if we have fewer combined points than in either band, something is wrong
            print(f'issue combining {supernova_name} g and r: fewer combined points than in either band')
            print(f'  ng={len(ddf_g["mag"].values)}, nr={len(ddf_r["mag"].values)}, nrg={x_all}')
            print(f'  last_nd_r={ndf_r["mjd"].max() if not ndf_r.empty else None}, last_nd_g={ndf_g["mjd"].max() if not ndf_g.empty else None}, last_nd_union={last_nd_union}')
        
        if x_all.size >= 2:  # need at least 2 points for the GP in check_candidates
            data_list_rg_row = check_candidates(
                oid=f'{supernova_name} (rg)', filt='rg', ifplot=PLOT, ifselfdata=True,
                epoch=x_all,
                mag=y_all,
                mag_err=e_all,
                non_epoch=non_epoch_rg,
                non_mag=non_mag_rg,
                last_nondet=[last_nd_union] if last_nd_union is not None else [],
                r_length=len(ddf_r['mag'].values),
                g_length=len(ddf_g['mag'].values),
                epoch_cut=200
            )
            # stash diagnostics about the combination
            data_list_rg_row['gr_offset_used'] = gr_off
            data_list_rg_row['gr_pairs_used'] = gr_npairs
        else:
            data_list_rg_row = {'oid': f'{supernova_name} (rg)'}  # minimal stub
    except Exception as _e:
        # keep going even if combine/fit fails
        data_list_rg_row = {'oid': f'{supernova_name} (rg)'}

    return supernova_name, data_list_r_row, data_list_g_row, data_list_rg_row


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    log('Running in parallel...')

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process, phot_files), total=len(phot_files)))

    log('Saving to dictionaries...')

    for name, r_result, g_result, rg_result in [r for r in results if r is not None]:
        if name is not None:
            data_list_r[name] = r_result
            data_list_g[name] = g_result
            data_list_rg[name] = rg_result  # NEW


    log('Merging r and g bands...')

    merged_data = []

    # Get all unique supernova names across both filters
    SN = set(data_list_r.keys()).union(data_list_g.keys()).union(data_list_rg.keys())

    for key in tqdm(SN):
        row = {'oid': key}
        
        if key in data_list_r:
            for k, v in data_list_r[key].items():
                if k == 'oid':
                    continue
                row[f"{k}_r"] = v
        
        if key in data_list_g:
            for k, v in data_list_g[key].items():
                if k == 'oid':
                    continue
                row[f"{k}_g"] = v

        if key in data_list_rg and data_list_rg[key] is not None:
            for k, v in data_list_rg[key].items():
                if k == 'oid':
                    continue
                row[f"{k}_rg"] = v
        
        merged_data.append(row)

    log('Saving to dataframe...')

    # Convert to DataFrame
    df = pd.DataFrame(merged_data)

    # Optional: Reorder columns for clarity
    cols = ['oid'] + sorted([col for col in df.columns if col != 'oid'])
    df = df[cols]

    log('Saving to CSV...')

    # Save to CSV
    output_path = os.path.join(DATA_DIR, f"{RUN_NAME}_interpretable_params{'_' + str(DAYS_AFTER) if DAYS_AFTER else ''}{'__' + str(DAYS_FROM_PEAK) if isinstance(DAYS_FROM_PEAK,(int,float)) else ''}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")