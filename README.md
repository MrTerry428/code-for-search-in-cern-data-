# code-for-search-in-cern-data-
search code
Here is the complete, ready-to-share VINES-IIB Analysis Toolkit v2.1 — fully equipped for open searches, including hidden-sector variants.


"""
VINES-IIB Warped Throat KK Graviton Search Toolkit
================================================
Author: Terry Vines (VINES Unified Field Theory Project)
Version: 2.1 (June 2026)
Purpose: Search for 1.6 TeV (and hidden) KK graviton in dilepton & diboson data
         Supports 1 / 3 / 5 vacua + hidden-sector suppressed couplings

Share freely with physicists, students, and independent researchers.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from iminuit import Minuit
from iminuit.cost import LeastSquares
import json
from pathlib import Path

# ====================== CORE VINES-IIB TEMPLATE ======================
def vines_kk_template(m, m_kk=1600.0, width=25.0, signal_strength=0.0,
                      background_scale=1e6, vacua_mode="3", 
                      channel="dilepton", hidden_mode=False):
    """
    VINES-IIB KK Graviton signal template
    hidden_mode: simulates suppressed coupling / broader width for hidden sector
    """
    # Background shape
    if channel == "dilepton":
        bg = background_scale * (m / 1000.0) ** (-3.8)
    else:  # diboson
        bg = background_scale * (m / 1000.0) ** (-2.9)
    
    # Width and suppression depending on vacua + hidden mode
    sigma_factor = 1.0 if vacua_mode == "1" else 1.05 if vacua_mode == "3" else 1.18
    if hidden_mode:
        sigma_factor *= 1.45   # broader for hidden sector
        signal_strength *= 0.35  # suppressed coupling
    
    signal = signal_strength * norm.pdf(m, loc=m_kk, scale=width * sigma_factor)
    return bg + signal


# ====================== MASS SCAN + FITTER ======================
def perform_vines_search(m_centers, observed, vacua_modes=None, 
                        hidden_search=False, channel="dilepton"):
    """Full search: fit at guessed mass + mass scan"""
    if vacua_modes is None:
        vacua_modes = ["1", "3", "5"]
    
    results = {}
    
    for mode in vacua_modes:
        for hidden in [False, hidden_search]:
            key = f"{mode}vacua_{'hidden' if hidden else 'standard'}"
            
            def model(m, sig_strength, bg_scale, m_kk=m_centers[np.argmax(observed)]):
                return vines_kk_template(m, m_kk=m_kk, signal_strength=sig_strength,
                                       background_scale=bg_scale, vacua_mode=mode,
                                       channel=channel, hidden_mode=hidden)
            
            cost = LeastSquares(m_centers, observed, 
                              np.sqrt(np.maximum(observed, 1)), model)
            
            m = Minuit(cost, sig_strength=100.0, bg_scale=1e6, m_kk=1600.0)
            m.limits["sig_strength"] = (0, None)
            m.limits["bg_scale"] = (1e5, None)
            m.limits["m_kk"] = (300, 5000)
            
            m.migrad()
            m.hesse()
            
            results[key] = {
                "m_kk": float(m.values["m_kk"]),
                "signal_strength": float(m.values["sig_strength"]),
                "signal_err": float(m.errors["sig_strength"]),
                "bg_scale": float(m.values["bg_scale"]),
                "fval": float(m.fval)
            }
    
    return results


# ====================== PLOTTING ======================
def plot_search_result(m_centers, observed, results, channel="dilepton"):
    plt.figure(figsize=(14, 9))
    plt.step(m_centers, observed, where='mid', color='black', lw=1.5, 
             label=f'Real/Simulated {channel} Data')
    
    colors = {'1vacua': 'red', '3vacua': 'orange', '5vacua': 'purple'}
    
    for key, res in results.items():
        mode = key.split('_')[0]
        style = '--' if 'hidden' in key else '-'
        col = colors.get(mode.replace('vacua','vacua'), 'blue')
        
        fit_total = vines_kk_template(m_centers,
                                    m_kk=res["m_kk"],
                                    signal_strength=res["signal_strength"],
                                    background_scale=res["bg_scale"],
                                    vacua_mode=mode[0],
                                    channel=channel,
                                    hidden_mode='hidden' in key)
        
        plt.plot(m_centers, fit_total, style, color=col, lw=2,
                 label=f"{key} @ {res['m_kk']:.1f} GeV (sig={res['signal_strength']:.1f})")
    
    plt.axvline(1600, color='red', ls=':', alpha=0.7, label='VINES Predicted 1.6 TeV')
    plt.xlabel(f'{channel.capitalize()} Reconstructed Mass [GeV]', fontsize=14)
    plt.ylabel('Events / bin (log)', fontsize=14)
    plt.yscale('log')
    plt.title(f'VINES-IIB Full Search — {channel.capitalize()} Channel', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'VINES_IIB_{channel}_FullSearch.png', dpi=300, bbox_inches='tight')
    plt.show()


# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    print("🚀 VINES-IIB Analysis Toolkit v2.1 - Ready for Run 2 + Run 3 Search")
    
    # === Replace this with REAL CERN histogram data ===
    m_bins = np.linspace(200, 6000, 400)
    m_centers = (m_bins[:-1] + m_bins[1:]) / 2
    
    # Example simulated data (replace with actual counts from ATLAS/CMS)
    data = vines_kk_template(m_centers, signal_strength=720, background_scale=9.2e5,
                           vacua_mode="5", channel="dilepton")
    observed = np.random.poisson(data)
    
    # === Run full search (standard + hidden) ===
    results = perform_vines_search(m_centers, observed, 
                                 vacua_modes=["1","3","5"], 
                                 hidden_search=True, 
                                 channel="dilepton")
    
    # Save results
    Path("vines_results").mkdir(exist_ok=True)
    with open("vines_results/dilepton_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== SEARCH RESULTS ===")
    for key, res in results.items():
        print(f"{key:25} → m_KK = {res['m_kk']:.1f} GeV | Signal = {res['signal_strength']:.1f} ± {res['signal_err']:.1f}")
    
    # Plot
    plot_search_result(m_centers, observed, results, channel="dilepton")
    
    print("\n✅ Analysis complete! Results saved to 'vines_results/' folder.")
    print("   Share this script freely. Look for the 1.6 TeV geometry!")
