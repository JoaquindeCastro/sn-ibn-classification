from lmfit import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import george
from george import kernels
from astropy.table import Table, unique
from scipy.stats import norm
from scipy import interpolate

from alerce.core import Alerce
client = Alerce()
from lmfit import Model

# --- Extrapolation controls (right side only) ---
EXTRAP_STEP_DAYS = 5.0       # grow the prediction window in 5-day steps
EXTRAP_MAX_DAYS_RIGHT = 30.0 # never extrapolate more than this many days
EXTRAP_SIGMA_MULT = 2.0      # accept extrapolated peak only if its GP σ <= 2× tail σ

# NEW MEAN TEMPLATE THINGY
_ibn = np.load("ibn_mean_r.npz")
IBN_GRID = _ibn["grid"]
IBN_MEAN = _ibn["mean_curve"]

mask_tmp = np.isfinite(IBN_GRID) & np.isfinite(IBN_MEAN)
IBN_GRID, IBN_MEAN = IBN_GRID[mask_tmp], IBN_MEAN[mask_tmp]
order = np.argsort(IBN_GRID)
IBN_GRID, IBN_MEAN = IBN_GRID[order], IBN_MEAN[order]


def get_LC(oid, filt='g', epoch_cut=200):
    if filt == 'g':
        fid = 1
    if filt == 'r':
        fid = 2

    mask_det = 0
    mask_nondet = 0
    
    SN_det = client.query_detections(oid, format='pandas')
    SN_nondet = client.query_non_detections(oid, format='pandas')
    
    if len(SN_det) > 0:
        SN_det = SN_det.sort_values("mjd")
        first_det = SN_det.mjd.values[0]
        mask_det = (SN_det.fid == fid) & (SN_det.mjd < (SN_det.mjd[0] + epoch_cut))
    if len(SN_nondet) > 0:
        SN_nondet = SN_nondet.sort_values("mjd")
        last_nondet = []
        last_nondet_index = SN_nondet.mjd < first_det
        if np.any(last_nondet_index):
            last_nondet = [SN_nondet[last_nondet_index].mjd.values[-1]]
            mask_nondet = (SN_nondet.fid == fid) & (SN_nondet.diffmaglim > -900) & (SN_nondet.mjd <= last_nondet[0])
        else:
            mask_nondet = []
    else:
        last_nondet = []
    
    if np.sum(mask_det) > 0:
        SN_det = SN_det[mask_det]
        time = SN_det['mjd']
        mag = SN_det['magpsf']
        mag_err = SN_det['sigmapsf']
        if np.sum(mask_nondet) > 0:
            SN_nondet = SN_nondet[mask_nondet]
            ref_index = SN_nondet.mjd < time.values[0]
            if np.all(SN_nondet.mjd >= time.values[0]):
                nodet_time = []
                nodet_mag = []
                #print ('no nondetection available from AleRCE for '+oid)
            else:
                #time = time - nodet_ref_time
                nodet_time = SN_nondet[ref_index].mjd
                nodet_mag = SN_nondet[ref_index].diffmaglim
                nodet_time = nodet_time.values
                nodet_mag = nodet_mag.values
        else:
            nodet_time = []
            nodet_mag = []
            #print ('no nondetection available from AleRCE for '+oid)
        return time.values, mag.values, mag_err.values, nodet_time, nodet_mag, last_nondet
    else:
        return [], [], [], [], [], last_nondet

def _delta_m_at(days_after, x_src, y_src, peak_epoch, peak_mag, z=None):
    """
    Δm at (peak + days_after). If z is given, interpret days_after in REST frame
    and convert to observed days via (1+z).
    Returns np.nan if x_ref is outside the coverage (no extrapolation).
    """
    # convert rest-frame → observed-frame time offset if redshift provided
    scale = (1.0 + z) if (z is not None) else 1.0
    x_ref = peak_epoch + days_after * scale

    if x_ref < np.min(x_src) or x_ref > np.max(x_src):
        return np.nan  # no coverage at that time

    y_ref = np.interp(x_ref, x_src, y_src)
    dm = y_ref - peak_mag  # mags increase as it fades
    return dm if dm >= 0 else -9999  # clamp tiny negatives from noise


def ibn_template(phase):
    """Template magnitude at a given phase (days since peak)."""
    # clamp outside coverage to edge values to avoid NaNs
    return np.interp(phase, IBN_GRID, IBN_MEAN, left=IBN_MEAN[0], right=IBN_MEAN[-1])

def check_candidates(oid = 'ZTF21aatisro', filt='g',
                     interp_epoch=-9999, ifplot=False, ifselfdata=False,
                     epoch=[], mag=[], mag_err=[], non_epoch=[], non_mag=[], last_nondet=[],
                     interp=True, epoch_cut=80, magdiff_cut=4,
                     extrapolate_rise=False, rise_slope_fixed_window=10, max_extrap_days_after_last=30,r_length=0,g_length=0
                     ):
    quality_flag = 1
    
    s0 = -9999
    rise_slope = -9999
    rise_slope_fixed = -9999
    decline_slope = -9999
    decline_slope_fixed = -9999
    decline_plot_x = None
    decline_plot_y = None
    dm15 = -9999
    rise_time = -9999
    peak_mag = -9999
    peak_epoch = -9999
    ndetection=0
    lc_params = {}
    lc_params['oid'] = oid
    lc_params['filt'] = filt
        
    if not ifselfdata:
        x, y, y_err, non_x, non_y, last_nondet = get_LC(oid, filt=filt, 
                                                 epoch_cut=epoch_cut)
        obj_info = client.query_object(oid, format='pandas')
        lc_params['meanra'] = obj_info['meanra'].values[0]
        lc_params['meandec'] = obj_info['meandec'].values[0]
        assert len(x) > 0, "There is no detection for "+oid+' in '+filt+' band'
        assert len(last_nondet) > 0, "There is no nondetection for "+oid
        assert x[0] - last_nondet[0] < 10, "last non detection is 10 days ago"
        
    else:
        x = epoch
        y = mag
        y_err = mag_err
        non_x = non_epoch
        non_y = non_mag
        last_nondet = last_nondet
        if len(x) > 0:
            if filt != 'rg':
                index = np.where(x-x[0] < epoch_cut)
                #assert len(index[0]) > 0, "No data available"
                x = x[index]
                y = y[index]
                y_err = y_err[index]
            else:
                # make sure we're working with numpy arrays
                x = np.asarray(x, float)
                y = np.asarray(y, float)
                y_err = np.asarray(y_err, float)
                non_x = np.asarray(non_x, float) if len(non_x) else non_x
                non_y = np.asarray(non_y, float) if len(non_y) else non_y

                # ---- choose a smarter anchor for the epoch window ----
                anchors = [x[0], max(x[-1] - epoch_cut, x[0])]  # first point, and trailing window

                # also try: first detection AFTER the last nondetection we were given
                if len(last_nondet):
                    try:
                        t_nd = float(last_nondet[0])
                        i = np.searchsorted(x, t_nd + 1e-6)
                        if i < len(x):
                            anchors.append(x[i])
                    except Exception:
                        pass

                # pick the anchor that keeps the most points inside [anchor, anchor+epoch_cut]
                def _count_in(lo):
                    hi = lo + epoch_cut
                    return np.count_nonzero((x >= lo) & (x <= hi))

                best_lo = max(anchors, key=_count_in)
                mask = (x >= best_lo) & (x <= best_lo + epoch_cut)
                x, y, y_err = x[mask], y[mask], y_err[mask]

                # also trim nondetections to be strictly before the kept window
                if len(non_x):
                    ndmask = non_x < x[0]
                    non_x = non_x[ndmask]
                    non_y = non_y[ndmask]

                lc_params['first_det'] = float(x[0]) if len(x) else -9999

        lc_params['meanra'] = None
        lc_params['meandec'] = None

    #if len(y_err) == 0:
    #    y_err = np.full(len(x), 0.05)
    if len(last_nondet)!=0:
        lc_params['last_nondet'] = last_nondet[0]
    else:
        lc_params['last_nondet'] = -9999
        
    #mag_mask = np.where(y < (np.min(y) + magdiff_cut))
    if len(x) > 0:
        # EDIT BROUGHT IT BACK CUS LOWKEY MIGHT HAVE KILLED PROGRESS; REMOVED THIS ON AUGUST 13 CUS OF MISSCLASSIFIED IBNS THAT WERE MISSING IMPT DATAPOINTS
        mag_mask = np.logical_or(x < x[np.where(y == np.min(y))[0][0]], y < (np.min(y) + magdiff_cut))
        x = x[mag_mask]
        #print(x)
        y = y[mag_mask]
        y_err = y_err[mag_mask]
        lc_params['first_det'] = x[0]
    else:
        lc_params['first_det'] = -9999
    
    if len(x) >= 2:
        lc_params['duration'] = x[-1] - x[0]
    else:
        lc_params['duration'] = -9999
        
    lc_params['ndetection'] = len(x)
    
    if len(x) >= 2 and (np.mean(np.diff(x)) < 0.001 or len(x) != len(np.unique(x))):#np.var(x) < 2.5e-7
        x, y, y_err = merge_photometry(x, y, y_err)
    
    if (len(x) !=0) and (len(non_x) != 0) and (x[0] - non_x[-1] > 0.1):
        s0 = (non_y[-1] - y[0])/(x[0] - non_x[-1])
    lc_params['s0'] = s0
    
    rise_time_flag = -9999
    if len(x) >= 3:
        #metric = max(np.var(x), 0.05)
        x_pred, y_pred, pred_var = GP_predict(x, y, y_err, metric=np.var(x))
        peak_epoch = x_pred[y_pred == np.min(y_pred)][0]
        peak_mag = np.min(y_pred)

        # FIT OFFSET AND MAGNITUDE SCALE TO THE TEMPLATE
        phase_obs = x - peak_epoch
        tmpl_obs  = ibn_template(phase_obs)

        A = np.vstack([np.ones_like(tmpl_obs), tmpl_obs]).T
        m0_hat, amp_hat = np.linalg.lstsq(A, y, rcond=None)[0]  # y equals-ish m0 + amp*template

        # Detrend with the mean and GP the residuals
        y_resid = y - (m0_hat + amp_hat * tmpl_obs)
        x_pred_resid, y_resid_pred, pred_var = GP_predict(x, y_resid, y_err, metric=np.var(x))

        # Final model = residual GP + (fitted mean at the same phases)
        x_pred = x_pred_resid
        y_pred = y_resid_pred + (m0_hat + amp_hat * ibn_template(x_pred - peak_epoch))

        # RECOMPUTE PEAK EPOCH AND PEAK MAG
        peak_epoch = x_pred[np.argmin(y_pred)]
        peak_mag   = np.min(y_pred)
        
        
        if len(last_nondet) != 0:
            if peak_epoch > x[0] and peak_epoch < x[-2]:
                rise_time = peak_epoch - last_nondet[0]
                rise_time_flag = 1
            elif peak_epoch < x[0]:
                _x = np.insert(x, 0, last_nondet[0])
                _y = np.insert(y, 0, 23) # artificial point, may need to change
                _y_err = np.insert(y_err, 0, 1)
                
                _x_pred, _y_pred, _pred_var = GP_predict(_x, _y, _y_err, metric=np.var(x))
                 
                peak_epoch = _x_pred[_y_pred == np.min(_y_pred)][0]
                peak_mag = np.min(_y_pred)
                
                rise_time_flag = 0 #rise_time is a limit
                #if peak_epoch > x[0]:
                #    rise_time = peak_epoch - last_nondet[0]
                #else:
                rise_time = x[0] - last_nondet[0]
            elif peak_epoch > x[-2]:
                # --- NEW: Extrapolate to the right to search for a plausible peak ---
                # Baseline tail uncertainty (median σ over last ~10% of in-window GP points)
                tail_n = max(10, len(pred_var)//10)
                tail_sigma = np.median(np.sqrt(pred_var[-tail_n:]))

                found = False
                # Step the right boundary outwards in EXTRAP_STEP_DAYS, up to EXTRAP_MAX_DAYS_RIGHT
                for pad in np.arange(EXTRAP_STEP_DAYS, EXTRAP_MAX_DAYS_RIGHT + 1e-6, EXTRAP_STEP_DAYS):
                    lo = np.min(x)
                    hi = x[-1] + float(pad)
                    xp2, yp2, vp2 = GP_predict(x, y, y_err, metric=np.var(x), x_min=lo, x_max=hi)

                    j = int(np.argmin(yp2))
                    tpk = float(xp2[j])
                    sig_pk = float(np.sqrt(vp2[j]))

                    # accept only if (a) not sitting on the hard right boundary and (b) uncertainty reasonable
                    eps = 1e-3
                    not_on_edge = (tpk < hi - eps)
                    ok_uncert = (sig_pk <= EXTRAP_SIGMA_MULT * tail_sigma)

                    if not_on_edge and ok_uncert and (tpk > x[0] + eps):
                        # We only change RISE TIME; we do NOT change peak_epoch used elsewhere.
                        rise_time = tpk - last_nondet[0]
                        if rise_time > 0:
                            rise_time_flag = 2  # 2 = extrapolated (right)
                            # keep peak_epoch unchanged so decline/rise slopes are unaffected
                            lc_params['rise_time_extrap'] = float(rise_time)
                            lc_params['rise_time_method'] = f'extrap_right_{pad:.0f}d'
                            found = True
                            break

                # If we never found a trustworthy min, leave rise_time as-is (no change).
                # (Optionally, you could fall back to a lower-limit here, but per request we only extrapolate.)

                    
        rise_index = np.where(x <= peak_epoch)
        if len(rise_index[0]) >= 2 and np.mean(np.diff(x[rise_index])) > 0.3: # possibly change to 1
            if interp and rise_time_flag != 0:
                
                x_rise, y_rise, pred_var_rise = GP_predict(x[rise_index], y[rise_index], y_err[rise_index], metric=np.var(x[rise_index]))
                rise_result = linear_fit(x_rise, y_rise)
            elif interp and rise_time_flag == 0:
                #x_rise, y_rise, pred_var_rise = GP_predict(_x[rise_index], _y[rise_index], _y_err[rise_index], metric=np.var(_x[rise_index]))
                #rise_result = linear_fit(x_rise, y_rise)
                _rise_index = np.logical_and(_x_pred>=lc_params['first_det'], _x_pred<=peak_epoch)
                rise_result = linear_fit(_x_pred[_rise_index], _y_pred[_rise_index])
            else:
                rise_result = linear_fit(x[rise_index], y[rise_index])
            rise_slope = -rise_result.params['a'].value

        decline_index = np.where(x >= peak_epoch)
        if len(decline_index[0]) >= 2 and np.mean(np.diff(x[decline_index])) > 0.01:
            #print (x_pred[np.where(x_pred >= peak_epoch)])
            if interp and rise_time_flag != 0:
                decline_result = linear_fit(x_pred[np.where(x_pred >= peak_epoch)], y_pred[np.where(x_pred >= peak_epoch)],
                                           a=0.1, b=-5000)
            elif interp and rise_time_flag == 0:
                decline_result = linear_fit(_x_pred[np.where(_x_pred >= peak_epoch)], _y_pred[np.where(_x_pred >= peak_epoch)],
                                           a=0.1, b=-5000)
            else:
                decline_result = linear_fit(x=x[decline_index], y=y[decline_index], 
                                                               a=0.1, b=-5000)
            decline_slope = decline_result.params['a'].value
        #print(rise_result.fit_report())
        #print(decline_result.fit_report())

        # NEW DECLINE SLOPE
        T1, T2 = 3.0, 18.0
        min_points = 2  # FOR A STABLE LINE

        def _weighted_linear_fit(xw, yw, w=None, a0=0.08):
            gmodel = Model(linear_fun)
            if w is None:
                return gmodel.fit(yw, x=xw, a=a0, b=np.median(yw))
            else:
                return gmodel.fit(yw, x=xw, a=a0, b=np.median(yw), weights=w)

        xp, yp, vp = x_pred, y_pred, pred_var
        # based on previous code
        if rise_time_flag == 0:
            xp, yp, vp = _x_pred, _y_pred, _pred_var
        lo, hi = peak_epoch + T1, min(peak_epoch + T2, xp.max())
        m = (xp >= lo) & (xp <= hi)
        decline_window = (lo, hi)

        if interp:
            # downweight uncertain tail by
            # 1) the window cut
            # 2) the uncertainty cut (uncertainty appears to be higher away from peak)

            if m.sum() < min_points:
                # widen adaptively until we have enough points, but never beyond the GP domain
                hi = min(peak_epoch + 25.0, xp.max())
                m = (xp >= peak_epoch + 0.5) & (xp <= hi)
            if m.sum() >= min_points:
                
                w = 1.0 / np.clip(vp[m], np.percentile(vp[m], 5), None)   # uncertainty weights
                res = _weighted_linear_fit(xp[m], yp[m], w=w, a0=0.08)
                a = res.params['a'].value
                b = res.params['b'].value
                decline_slope_fixed = a
                # save line segment for plotting
                decline_plot_x = xp[m]
                decline_plot_y = linear_fun(decline_plot_x, a, b)

        dm15 = _delta_m_at(15.0, xp, yp, peak_epoch, peak_mag, z=None)

        '''else:
            # Fit on observed points in the same early window
            lo, hi = peak_epoch + T1, peak_epoch + T2
            m = (x >= lo) & (x <= hi)
            if m.sum() < min_points:
                hi = min(peak_epoch + 25.0, x.max())
                m = (x >= peak_epoch + 0.5) & (x <= hi)
            if m.sum() >= min_points:
                w = 1.0 / np.clip(y_err[m], 0.02, None)     # photometric weights
                res = _weighted_linear_fit(x[m], y[m], w=w, a0=0.08)
                decline_slope = res.params['a'].value'''


    elif len(x) == 2 and np.mean(np.diff(x)) > 0.3:
        result = linear_fit(x, y)
        slope = result.params['a'].value
        if slope < 0:
            rise_index = np.where(x > -100)
            rise_slope = -slope
            rise_result = linear_fit(x, y)
        else:
            decline_index = np.where(x > -100)
            decline_slope = slope
            decline_slope_fixed = slope # FALL BACK
            decline_result = linear_fit(x, y)
    
    window_start = peak_epoch - rise_slope_fixed_window
    window_mask = (x >= window_start) & (x <= peak_epoch)
    if np.sum(window_mask) >= 2:
        x_window = x[window_mask]
        y_window = y[window_mask]
        try:
            slope = np.polyfit(x_window, y_window, 1)[0]
            rise_slope_fixed = -slope  # negative sign for brightening
        except Exception:
            rise_slope_fixed = -9999
    else:
        rise_slope_fixed = -9999
    
    lc_params['rise_time'] = rise_time
    lc_params['peak_mag'] = peak_mag
    lc_params['peak_epoch'] = peak_epoch
    lc_params['rise_time_flag'] = rise_time_flag
    lc_params['rise_slope'] = rise_slope
    lc_params['decline_slope'] = decline_slope
    lc_params['decline_slope_fixed'] = decline_slope_fixed
    lc_params['dm15']  = float(dm15)  if np.isfinite(dm15)  else -9999
    
    if ifplot:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,6.18))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        params = {'figure.max_open_warning': 50,
                 'legend.fontsize': 18,
                  'figure.figsize': (15, 5),
                 'axes.labelsize': 15,
                 'axes.titlesize':20,
                 'xtick.labelsize':12,
                 'ytick.labelsize':12,
                 'ytick.major.size': 5.5,
                 'axes.linewidth': 2}
        plt.rcParams.update(params)
        ax = axs
        
        #ax.plot(x, y, 'bo')
        ax.errorbar(x, y, yerr=y_err, fmt="bo", capsize=0)
        if len(last_nondet) != 0:
            ax.axvline(last_nondet, label='last nondet', color='blue', alpha=0.3)
        if len(non_x) != 0:
            ax.plot(non_x, non_y, 'bs', alpha=0.3)
        #plt.plot(x, result.init_fit, 'k--', label='initial fit')
        if len(x) >= 3:
            ax.axvline(peak_epoch, 
                           label='Gaussian process peak', color='red')
            if rise_time_flag == 1 or rise_time_flag == -9999:
                ax.plot(x_pred, y_pred, 'r-', label='Gaussian process fit')
                ax.fill_between(x_pred, y_pred - np.sqrt(pred_var), y_pred + np.sqrt(pred_var),
                color="k", alpha=0.2)
                #ax.plot(x_rise, y_rise, 'g-', label='Gaussian process rise')
            if rise_time_flag == 0:
                ax.plot(_x_pred, _y_pred, 'r-', label='Gaussian process fit (add non)')
                ax.fill_between(_x_pred, _y_pred - np.sqrt(_pred_var), _y_pred + np.sqrt(_pred_var),
                color="k", alpha=0.2)
        if rise_slope!=-9999:
            ax.plot(x[rise_index], linear_fun(x[rise_index], *rise_result.values.values()),
                    'k-', label='rise')
        if decline_slope!=-9999:
            ax.plot(x[decline_index], linear_fun(x[decline_index], *decline_result.values.values()),
                    'g-', label='decline')
        if decline_slope!=-9999:
            ax.plot(x[decline_index], linear_fun(x[decline_index], *decline_result.values.values()),
                    'g-', label='decline')
        # PLOT NEW FIXED DECLINE SLOPE
        if decline_plot_x is not None and len(decline_plot_x) > 1 and decline_slope_fixed != -9999:
            ax.plot(decline_plot_x, decline_plot_y, 'orange', lw=1,
                    label=f'decline (weighted, +{T1:.0f}→+{T2:.0f} d)')
            # light shading for the intended window (may be wider if fallback used)
            ax.axvspan(float(decline_window[0]), float(decline_window[1]),
                    color='orange', alpha=0.05)
            # annotate slope value
            ax.text(decline_plot_x.mean(), np.interp(decline_plot_x.mean(), decline_plot_x, decline_plot_y),
                    f"{decline_slope_fixed:.3f} mag/day", color='orange', fontsize=10,
                    ha='center', va='bottom')
        if s0!=-9999:
            ax.plot([non_x[-1], x[0]], [non_y[-1], y[0]], 'y-', label='s0')
        ax.legend()
        ax.set_title(oid)
        #ax.set_xlim(0, 1)
        try:
            ax.set_ylim(min(y)-0.5, max(y)+0.5)
        except ValueError:
            print('Error with ax.set_ylim(min(y)-0.5, max(y)+0.5)')
        ax.set_xlabel('epoch', size=20)
        ax.set_ylabel('magnitude', size=20)
        ax.minorticks_on()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 15, length = 8, width = 2)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 12, length = 4, width = 1)
        #ax.set_xlim(min(x)-0.01, max(x)+0.01)
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.show()
        plt.close(fig) # added these two lines like in Alex' code to close figures
    return lc_params

def check_candidates_color(oid = 'ZTF21aatisro',
                     interp_epoch=-9999, ifplot=False, ifsndavis=False, ftb=False, ifselfdata=False,
                     epoch_g=[], mag_g=[], mag_err_g=[], non_epoch_g=[], non_mag_g=[], 
                     epoch_r=[], mag_r=[], mag_err_r=[], non_epoch_r=[], non_mag_r=[], 
                     last_nondet=[],
                     interp=True, epoch_cut=40, magdiff_cut=1.5):
    lc_params = {}
    lc_params['oid'] = oid
        
    if not ifselfdata:
        if ftb:
            x_g, y_g, y_err_g, non_x_g, non_y_g, last_nondet = get_LC_ftb(oid, filt='g', 
                                                 epoch_cut=epoch_cut)
            x_r, y_r, y_err_r, non_x_r, non_y_r, last_nondet = get_LC_ftb(oid, filt='r', 
                                                 epoch_cut=epoch_cut)
            lc_params['meanra'] = None
            lc_params['meandec'] = None
        elif ifsndavis:
            x_g, y_g, y_err_g, non_x_g, non_y_g, last_nondet = get_LC_sndavis(oid, filt='g', 
                                                 epoch_cut=epoch_cut)
            x_r, y_r, y_err_r, non_x_r, non_y_r, last_nondet = get_LC_sndavis(oid, filt='r', 
                                                 epoch_cut=epoch_cut)
            lc_params['meanra'] = None
            lc_params['meandec'] = None
        else:
            x_g, y_g, y_err_g, non_x_g, non_y_g, last_nondet = get_LC(oid, filt='g', 
                                                 epoch_cut=epoch_cut)
            x_r, y_r, y_err_r, non_x_r, non_y_r, last_nondet = get_LC(oid, filt='r', 
                                                 epoch_cut=epoch_cut)
            obj_info = client.query_object(oid, format='pandas')
            lc_params['meanra'] = obj_info['meanra'].values[0]
            lc_params['meandec'] = obj_info['meandec'].values[0]
        #assert len(x) > 0, "There is no detection for "+oid+' in '+filt+' band'
        #assert len(last_nondet) > 0, "There is no nondetection for "+oid
        #assert x[0] - last_nondet[0] < 10, "last non detection is 10 days ago"
        
    else:
        x_g = epoch_g
        y_g = mag_g
        y_err_g = mag_err_g
        non_x_g = non_epoch_g
        non_y_g = non_mag_g
        x_r = epoch_r
        y_r = mag_r
        y_err_r = mag_err_r
        non_x_r = non_epoch_r
        non_y_r = non_mag_r
        last_nondet = last_nondet
        if len(x_g) > 0:
            index = np.where(x_g-x_g[0] < epoch_cut)
            #assert len(index[0]) > 0, "No data available"
            x_g = x_g[index]
            y_g = y_g[index]
            y_err_g = y_err_g[index]
            index = np.where(x_r-x_r[0] < epoch_cut)
            #assert len(index[0]) > 0, "No data available"
            x_r = x_r[index]
            y_r = y_r[index]
            y_err_r = y_err_r[index]
        lc_params['meanra'] = None
        lc_params['meandec'] = None
    
    #if len(y_err) == 0:
    #    y_err = np.full(len(x), 0.05)
    if len(last_nondet)!=0:
        lc_params['last_nondet'] = last_nondet[0]
    else:
        lc_params['last_nondet'] = -9999
    if len(x_g) >= 2 and (np.mean(np.diff(x_g)) < 0.001 or len(x) != len(np.unique(x_g))):#np.var(x) < 2.5e-7
        x_g, y_g, y_err_g = merge_photometry(x_g, y_g, y_err_g)
    if len(x_r) >= 2 and (np.mean(np.diff(x_r)) < 0.001 or len(x) != len(np.unique(x_r))):#np.var(x) < 2.5e-7
        x_r, y_r, y_err_r = merge_photometry(x_r, y_r, y_err_r)
    color_info = {}
    color_info['last_nondet'] = last_nondet
    color_info['phase'] = []
    color_info['g-r'] = []
    if len(x_g) == 1 and len(x_r) == 1:
        if x_g - x_r < 3:
            color_info['phase'] = (x_g + x_r)/2.
            color_info['g-r'] = y_g - y_r
    elif len(x_g) >= 1 \
        and len(x_r) >= 1 :
        if len(x_g) >= len(x_r):
            x_pred, y_pred, pred_var = GP_predict(x_g, y_g, y_err_g, metric=np.var(x_g))
            fmodel = interpolate.interp1d(x_pred, y_pred, bounds_error=False)
            y_g = fmodel(x_r)
            #x_g = x_r
            phase_left = np.max([np.min(x_g), np.min(x_r)]) - 1.
            phase_right = np.min([np.max(x_g), np.max(x_r)]) + 1.
            color_info['phase'] = x_r
            color_info['g-r'] = y_g - y_r
            phase_index = (color_info['phase'] > phase_left) & (color_info['phase'] < phase_right)
            color_info['phase'] = color_info['phase'][phase_index]
            color_info['g-r'] = color_info['g-r'][phase_index]
        elif len(x_g) < len(x_r):
            x_pred, y_pred, pred_var = GP_predict(x_r, y_r, y_err_r, metric=np.var(x_r))
            fmodel = interpolate.interp1d(x_pred, y_pred, bounds_error=False)
            y_r = fmodel(x_g)
            #x_r = x_g
            phase_left = np.max([np.min(x_g), np.min(x_r)]) - 1.
            phase_right = np.min([np.max(x_g), np.max(x_r)]) + 1.
            color_info['phase'] = x_g
            color_info['g-r'] = y_g - y_r
            phase_index = (color_info['phase'] > phase_left) & (color_info['phase'] < phase_right)
            color_info['phase'] = color_info['phase'][phase_index]
            color_info['g-r'] = color_info['g-r'][phase_index]

            

    
    if ifplot:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,6.18))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        params = {'figure.max_open_warning': 50,
                 'legend.fontsize': 18,
                  'figure.figsize': (15, 5),
                 'axes.labelsize': 15,
                 'axes.titlesize':20,
                 'xtick.labelsize':12,
                 'ytick.labelsize':12,
                 'ytick.major.size': 5.5,
                 'axes.linewidth': 2}
        plt.rcParams.update(params)
        ax = axs
        
        
        if len(last_nondet) != 0:
            ax.plot(color_info['phase']-last_nondet, color_info['g-r'], 'bo')
            ax.axvline(0, label='last nondet', color='blue', alpha=0.3)
        else:
            ax.plot(color_info['phase']-np.min(color_info['phase']), color_info['g-r'], 'bo')
        ax.legend()
        ax.set_title(oid)
        #ax.set_xlim(0, 1)
        ax.set_ylim(min(color_info['g-r'])-0.5, max(color_info['g-r'])+0.5)
        ax.set_xlabel('epoch', size=20)
        ax.set_ylabel('g-r', size=20)
        ax.minorticks_on()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 15, length = 8, width = 2)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 12, length = 4, width = 1)
        #ax.set_xlim(min(x)-0.01, max(x)+0.01)
        #ax.set_ylim(ax.get_ylim()[::-1])
    return color_info

def GP_predict(x, y, y_err, metric, nsample=[],x_min=None,x_max=None):
    # CHANGE THIS TO USE NON_INTERPRETABLE GP FIT
    # rebuild by filter 
    x = np.asarray(x)
    y = np.asarray(y)
    y_err = np.asarray(y_err)

    # Remove entries with NaN or inf
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
    x, y, y_err = x[mask], y[mask], y_err[mask]
    # CONVERT TO FLUX
    # CHANGE TO SOMETHING LIKE CHECK CANDIDATES COLOR

    # Early exit if not enough data
    if len(x) < 2:
        raise ValueError("Too few valid points for GP")
    
    #kernel = np.var(y) * kernels.ExpSquaredKernel(metric)+np.mean(y)
    # keep kernel strictly positive, put mean in the GP (not in the kernel)
    metric = float(metric) if np.isfinite(metric) and metric > 0 else max(np.ptp(x), 1.0)
    kernel = np.var(y) * kernels.ExpSquaredKernel(metric)
    gp = george.GP(kernel, mean=float(np.nanmean(y)))
    gp.compute(x, y_err + 1e-6)  # tiny jitter for stability

    #y0 = y - np.mean(y)
    #kernel = kernels.ConstantKernel(np.var(y0)) * kernels.ExpSquaredKernel(metric)
    gp = george.GP(kernel)
    gp.compute(x, y_err)

    # NEW: allow explicit prediction range (for rightward extrapolation)
    lo = np.min(x) if x_min is None else float(x_min)
    hi = np.max(x) if x_max is None else float(x_max)
    if hi <= lo:  # safety
        hi = lo + 1e-3
        
    #x_pred = np.linspace(np.min(x), np.max(x), int((np.max(x)-np.min(x))/0.005))
    #x_pred = np.arange(np.min(x)-0.1, np.max(x)+0.1, 0.005)

    if len(nsample) == 0:
        x_pred = np.linspace(lo, hi, max(int((hi - lo)/0.005), 500))
    else:
        x_pred = np.linspace(lo, hi, nsample[0])

    y_pred, pred_var = gp.predict(y, x_pred, return_var=True)
    return x_pred, y_pred, pred_var

    if len(nsample) == 0:
        x_pred = np.linspace(np.min(x), np.max(x), max(int((np.max(x)-np.min(x))/0.005), 500))
    else:
        x_pred = np.linspace(np.min(x), np.max(x), nsample[0])
    y_pred, pred_var = gp.predict(y, x_pred, return_var=True)
    # CONVERT BACK TO MAG
    return x_pred, y_pred, pred_var


def skew_gaussian(x, amp, cen, wid, y0, a):
    return amp * norm.pdf((x-cen)/wid) * norm.cdf(a*(x-cen)/wid) + y0

def simulate_LC(cadence = 1., sigma=0.05, parameters={'amp': -4.35312553, 
                                          'cen':0.61253628, 
                                          'wid':5.11074593, 
                                          'y0':21.5884369, 
                                          'a':9.99831793},
               rand_num = 1000):
    starting_time = np.random.uniform(-15,-5,rand_num)
    fake_lc = []
    for _starting_time in starting_time:
        all_time = np.arange(_starting_time, 20, cadence)
        all_mag = skew_gaussian(all_time, *parameters.values())
        index1 = np.where(all_time > 0)
        time = all_time[index1]
        mag = all_mag[index1] 
        mag = mag + np.random.normal(0, sigma, len(mag))
        mag_err = np.full(len(mag), sigma)
        index2 = np.where(all_time <= 0)
        non_time  = all_time[index2]
        non_mag = all_mag[index2]
        _fake_lc = {'time':time,
                    'mag':mag,
                    'mag_err':mag_err,
                    'non_time':non_time,
                    'non_mag':non_mag}
        fake_lc.append(_fake_lc)
    return fake_lc

def detection_time_series(epoch, mag, mag_err, nodet_time, nodet_mag, last_nondet, obs_epoch=[], ifplot_sep=False, ifplot_sum=True):

    #simulator_percen = np.zeros((len(obs_time),4))
    if len(obs_epoch) == 0:
        simulator_result = np.full((len(epoch),4), -9999.)
        for index, _epoch in enumerate(epoch):
            epoch_mask = np.where(epoch <= _epoch)
            result = check_candidates(oid='{} detection'.format(index+1), epoch=epoch[epoch_mask], mag=mag[epoch_mask], mag_err=mag_err[epoch_mask], non_epoch=nodet_time, non_mag=nodet_mag, last_nondet=last_nondet, ifplot=ifplot_sep, ifselfdata=True, interp=True)
            simulator_result[index, :] = [result['s0'], result['rise_time'], 
                                          result['rise_slope'], result['decline_slope']]
    else:
        simulator_result = np.full((len(obs_epoch),4), -9999.)
        for index, _epoch in enumerate(obs_epoch):
            epoch_mask = np.where(epoch <= _epoch)
            result = check_candidates(oid='{} detection'.format(index+1), epoch=epoch[epoch_mask], mag=mag[epoch_mask], mag_err=mag_err[epoch_mask], non_epoch=nodet_time, non_mag=nodet_mag, last_nondet=last_nondet, ifplot=ifplot_sep, ifselfdata=True, interp=True)
            simulator_result[index, :] = [result['s0'], result['rise_time'], 
                                          result['rise_slope'], result['decline_slope']]
    return simulator_result

def linear_fun(x, a, b):
    return a * x + b
def linear_fit(x, y, a=-0.1, b=5000):
    gmodel = Model(linear_fun)
    result = gmodel.fit(y, x=x, a=a, b=b)
    return result

def merge_photometry(x, y, y_err):
    x = np.array(x)
    y = np.array(y)
    y_err = np.array(y_err)
    
    x = np.round(x, 3)
    x, index, inverse, counts = np.unique(x, return_index=True, return_inverse=True, return_counts=True)
    duplicate = np.where(counts > 1)[0]
    #print (duplicate)
    dup_index = index[duplicate]
    #print (dup_index)
    dup_counts = counts[duplicate]
    for i, var in enumerate(dup_index):
        var = int(var)
        y[var], y_err[var] = weighted_average(y[var: var+dup_counts[i]], y_err[var: var+dup_counts[i]])
    #print (index)
    y = y[index]
    y_err = y_err[index]
    return x, y, y_err
def weighted_average(y, y_err):
    y_bar = np.sum(y/y_err**2.)/np.sum(1/y_err**2.)
    y_err_bar = np.sqrt(1/np.sum(1/y_err**2))
    return y_bar, y_err_bar

def result_process(result_all):
    result_merge = {}
    for key in result_all[0].keys():
        _value = []
        for result in result_all:
            _value.append(result[key])
        result_merge[key] = _value
    return result_merge