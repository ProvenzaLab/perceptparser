import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import specparam


def find_psd_pickle() -> Path:
    candidates = [
        Path("psd_rs.pkl"),
        Path(__file__).resolve().parent / "psd_rs.pkl",
        Path(__file__).resolve().parents[1] / "psd_rs.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find psd_rs.pkl. Run combine_data.py first to create it, "
        "or place psd_rs.pkl next to run_neural_cv.py or at repository root."
    )


psd_pickle = find_psd_pickle()
with open(psd_pickle, "rb") as f:
    psd_rs = pickle.load(f)


psds_left = psd_rs["psds_left"]
psds_right = psd_rs["psds_right"]
madrs = psd_rs["madrs"]
nbu_visits = psd_rs["NBU_visits"]
channel_name_left = psd_rs["channel_name_left"]
channel_name_right = psd_rs["channel_name_right"]


fit_range = [1, 40]
plot_xlim = [0, 40]
output_pdf = "psd_specparam_fits_0_40Hz.pdf"
output_feature_csv = "psd_specparam_features_0_40Hz.csv"

bands = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "low_beta": (13, 20),
    "high_beta": (20, 35),
}


def band_means(freqs: np.ndarray, spectrum: np.ndarray, band_definitions: dict) -> dict:
    values = {}
    for band_name, (f_low, f_high) in band_definitions.items():
        mask = (freqs >= f_low) & (freqs <= f_high)
        values[band_name] = float(np.nanmean(spectrum[mask])) if np.any(mask) else np.nan
    return values


feature_rows = []

with PdfPages(output_pdf) as pdf:
    for hemisphere, psd_dict, channel_dict in [
        ("left", psds_left, channel_name_left),
        ("right", psds_right, channel_name_right),
    ]:
        for date in sorted(psd_dict.keys()):
            freqs, psd = psd_dict[date]

            fm = specparam.SpectralModel(verbose=False)
            fm.fit(freqs, psd, fit_range)

            fit_freqs = fm.data.freqs
            original_log_power = fm.data.power_spectrum
            modeled_log_power = fm.results.model.get_component("full", space="log")
            aperiodic_log_power = fm.results.model.get_component("aperiodic", space="log")
            aperiodic_linear_power = fm.results.model.get_component("aperiodic", space="linear")

            raw_linear_power = np.interp(fit_freqs, freqs, psd)
            ap_corrected_linear_power = raw_linear_power - aperiodic_linear_power

            raw_band_values = band_means(fit_freqs, raw_linear_power, bands)
            ap_band_values = band_means(fit_freqs, ap_corrected_linear_power, bands)

            feature_row = {
                "date": date,
                "hemisphere": hemisphere,
                "madrs": madrs.get(date, np.nan),
                "nbu_visit": nbu_visits.get(date, np.nan),
                "channel_name": channel_dict.get(date, "NA"),
                "aperiodic_offset": float(fm.results.get_params("aperiodic", "offset")),
                "aperiodic_exponent": float(fm.results.get_params("aperiodic", "exponent")),
            }
            for band_name, value in raw_band_values.items():
                feature_row[f"raw_{band_name}"] = value
            for band_name, value in ap_band_values.items():
                feature_row[f"ap_{band_name}"] = value
            feature_rows.append(feature_row)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(fit_freqs, original_log_power, color="black", linewidth=1.25, label="Original spectrum")
            ax.plot(fit_freqs, modeled_log_power, color="tab:blue", linewidth=1.5, label="Model fit")
            ax.plot(fit_freqs, aperiodic_log_power, color="tab:red", linestyle="--", linewidth=1.5, label="Aperiodic")
            ax.set_xlim(plot_xlim)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Log10 Power")
            ax.set_title(
                f"{date} | {hemisphere} | MADRS={madrs.get(date, 'NA')} | "
                f"NBU={nbu_visits.get(date, 'NA')} | Channel={channel_dict.get(date, 'NA')}"
            )
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

df_features = pd.DataFrame(feature_rows)
df_features = df_features.sort_values(["date", "hemisphere"]).reset_index(drop=True)
df_features.to_csv(output_feature_csv, index=False)

print(f"Loaded PSDs from: {psd_pickle}")
print(f"Saved: {output_pdf}")
print(f"Saved: {output_feature_csv}")

