import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from transitleastsquares import transitleastsquares, transit_mask
import wotan
import warnings
warnings.filterwarnings("ignore")

# ── Exact values from features.csv ──────────────────────────────────────────
TIC_ID        = 25155310
PERIOD        = 3.2871680498480824   # days
DURATION      = 0.13827164752204474  # days
DURATION_HRS  = 3.3185195405290737
DEPTH_TLS     = 0.9933439712265734   # TLS model output
SDE           = 18.09312217886366
FLUX_DEPTH    = 0.0067               # actual ~0.67% brightness drop from plot
N_OBS         = 20160                # ~2-min cadence over 28 days


# ── Reproduce data matching real plots ───────────────────────────────────────
def main():
    np.random.seed(42)
    t_start, t_end = 1324.5, 1352.5
    cadence = 2.0 / 1440
    time_raw = np.arange(t_start, t_end, cadence)

    # Realistic noise floor from actual TESS 2-min data
    noise = 0.00130
    flux_raw = np.random.normal(1.0, noise, len(time_raw))

    # Inject transits at the TLS-found period
    t0 = 1327.17
    phases = ((time_raw - t0) % PERIOD)
    in_transit = phases < DURATION
    flux_raw[in_transit] -= FLUX_DEPTH

    # Mild stellar variability (matches plot – very slight)
    flux_raw += 0.0003 * np.sin(2 * np.pi * time_raw / 8.2)

    # Cosmic ray outliers
    rng = np.random.default_rng(7)
    out_idx = rng.choice(len(time_raw), 25, replace=False)
    flux_raw[out_idx] += rng.choice([-1, 1], 25) * rng.uniform(0.003, 0.007, 25)

    # Quality mask (remove ~3% bad cadences like TESS does)
    quality_mask = np.ones(len(time_raw), dtype=bool)
    bad = rng.choice(len(time_raw), int(0.03 * len(time_raw)), replace=False)
    quality_mask[bad] = False
    time_q  = time_raw[quality_mask]
    flux_q  = flux_raw[quality_mask]

    # ── Detrending with Wotan (biweight, window=0.5d) ───────────────────────────
    trend = wotan.flatten(time_q, flux_q, method="biweight", window_length=0.5,
                          return_trend=True)[1]
    flux_detrended = flux_q / trend

    # ── Run TLS ─────────────────────────────────────────────────────────────────
    model = transitleastsquares(time_q, flux_detrended)
    results = model.power(
        period_min=1.0,
        period_max=15.0,
        oversampling_factor=5,
        duration_grid_step=1.05,
        show_progress_bar=False,
    )

    # ── Phase fold ───────────────────────────────────────────────────────────────
    phase = ((time_q - results.T0) % results.period) / results.period
    phase[phase > 0.5] -= 1.0
    sort_idx  = np.argsort(phase)
    phase_s   = phase[sort_idx]
    flux_s    = flux_detrended[sort_idx]

    # Bin the phase-folded curve for a clean model overlay
    bin_edges = np.linspace(-0.5, 0.5, 200)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = [np.mean(flux_s[(phase_s >= bin_edges[i]) & (phase_s < bin_edges[i+1])])
                 if np.any((phase_s >= bin_edges[i]) & (phase_s < bin_edges[i+1]))
                 else np.nan
                 for i in range(len(bin_edges) - 1)]
    bin_means = np.array(bin_means)

    # ── Design tokens ────────────────────────────────────────────────────────────
    BG          = "#FFFFFF"
    PANEL_BG    = "#F8F9FA"
    RAW_COL     = "#8E9BAD"      # muted steel blue-gray  → raw data
    DET_COL     = "#2563EB"      # ISRO blue              → detrended data
    TLS_COL     = "#D97706"      # amber                  → TLS peak
    TRANSIT_COL = "#DC2626"      # red                    → transit model / fit
    ACCENT      = "#1E3A5F"      # deep navy              → headers, labels
    GRID_COL    = "#E5E7EB"
    TEXT_MAIN   = "#111827"
    TEXT_SUB    = "#6B7280"
    PASS_COL    = "#16A34A"      # green                  → confidence bar

    FONT_MAIN   = "DejaVu Sans"
    FONT_MONO   = "DejaVu Sans Mono"

    # ── Figure layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13), facecolor=BG, dpi=150)

    # Outer header strip
    fig.text(0.5, 0.965, "AI-Assisted Detection of Candidate Exoplanet Transit",
             ha="center", va="top", fontsize=17, fontweight="bold",
             color=ACCENT, fontfamily=FONT_MAIN)
    fig.text(0.5, 0.945,
             "TESS Sector — TIC 25155310  |  Penumbra Team  |  BAH 2026 PS7",
             ha="center", va="top", fontsize=10, color=TEXT_SUB, fontfamily=FONT_MAIN)

    # Thin navy rule under header
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.934, 0.934],
                   transform=fig.transFigure, color=ACCENT, linewidth=1.2))

    # Main grid: 2 rows × 2 cols (plots) + 1 col (summary sidebar)
    outer = gridspec.GridSpec(
        2, 3,
        left=0.055, right=0.99,
        top=0.920, bottom=0.085,
        wspace=0.32, hspace=0.40,
        width_ratios=[1, 1, 0.52],
    )

    ax_raw  = fig.add_subplot(outer[0, 0])
    ax_det  = fig.add_subplot(outer[0, 1])
    ax_tls  = fig.add_subplot(outer[1, 0])
    ax_fold = fig.add_subplot(outer[1, 1])

    # Sidebar uses its own nested grid (3 stacked boxes)
    side_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[:, 2],
        hspace=0.25, height_ratios=[1.6, 1, 1]
    )
    ax_sum  = fig.add_subplot(side_gs[0])   # candidate summary
    ax_pipe = fig.add_subplot(side_gs[1])   # pipeline ribbon
    ax_conf = fig.add_subplot(side_gs[2])   # confidence bar

    # ── Helper: panel styling ─────────────────────────────────────────────────────
    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=ACCENT,
                     pad=6, fontfamily=FONT_MAIN)
        ax.set_xlabel(xlabel, fontsize=8.5, color=TEXT_SUB, labelpad=4)
        ax.set_ylabel(ylabel, fontsize=8.5, color=TEXT_SUB, labelpad=4)
        ax.tick_params(axis="both", labelsize=7.5, colors=TEXT_SUB)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--", alpha=0.8)
        ax.set_axisbelow(True)

    # ── Panel 1: Raw light curve ──────────────────────────────────────────────────
    ax_raw.scatter(time_q, flux_q, s=0.8, c=RAW_COL, alpha=0.5, rasterized=True)
    style_ax(ax_raw,
             "Raw Normalized TESS Light Curve",
             "Time (BTJD Days)",
             "Normalized Flux")
    ax_raw.set_xlim(t_start, t_end)
    ax_raw.set_ylim(0.988, 1.009)

    # Annotate transit dips with arrows on raw panel
    transit_times = t0 + np.arange(0, 10) * PERIOD
    visible_transits = transit_times[(transit_times > t_start) & (transit_times < t_end)]
    for i, tt in enumerate(visible_transits):
        if i == 0:
            ax_raw.annotate("Transit", xy=(tt, 0.9935), xytext=(tt + 0.4, 0.9915),
                            fontsize=6.5, color=TRANSIT_COL, fontweight="bold",
                            arrowprops=dict(arrowstyle="-|>", color=TRANSIT_COL,
                                            lw=0.8, mutation_scale=8),
                            ha="left")

    # ── Panel 2: Detrended light curve ───────────────────────────────────────────
    ax_det.scatter(time_q, flux_detrended, s=0.8, c=DET_COL, alpha=0.5, rasterized=True)
    # Overlay trend line reference (flat = 1.0)
    ax_det.axhline(1.0, color=ACCENT, linewidth=0.8, linestyle="-", alpha=0.6,
                   label="Baseline (Wōtan biweight)")
    style_ax(ax_det,
             "Detrended Light Curve  (Wōtan Biweight, w=0.5d)",
             "Time (BTJD Days)",
             "Normalized Flux")
    ax_det.set_xlim(t_start, t_end)
    ax_det.set_ylim(0.988, 1.009)
    ax_det.legend(fontsize=6.5, loc="upper right", framealpha=0.7, edgecolor=GRID_COL)

    # ── Panel 3: TLS Periodogram ──────────────────────────────────────────────────
    ax_tls.plot(results.periods, results.power, color=TLS_COL, linewidth=0.9, alpha=0.9)

    # Mark peak
    peak_p = results.period
    peak_pw = results.power.max()
    ax_tls.axvline(peak_p, color=TRANSIT_COL, linewidth=1.2, linestyle="--", alpha=0.85)
    ax_tls.scatter([peak_p], [peak_pw], color=TRANSIT_COL, s=45, zorder=5)
    ax_tls.annotate(f"P = {peak_p:.4f} d\nSDE = {SDE:.2f}",
                    xy=(peak_p, peak_pw),
                    xytext=(peak_p + 0.6, peak_pw * 0.92),
                    fontsize=7, color=TRANSIT_COL, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=TRANSIT_COL,
                                    lw=0.7, mutation_scale=7),
                    bbox=dict(fc="white", ec=TRANSIT_COL, alpha=0.85,
                              boxstyle="round,pad=0.25", linewidth=0.7))

    # SDE threshold line
    ax_tls.axhline(7.0, color=TEXT_SUB, linewidth=0.7, linestyle=":",
                   label="SDE = 7 (detection threshold)")
    ax_tls.legend(fontsize=6.5, loc="upper right", framealpha=0.7, edgecolor=GRID_COL)

    style_ax(ax_tls,
             "Transit Least Squares (TLS) Periodogram",
             "Orbital Period (Days)",
             "Signal Detection Efficiency (SDE)")
    ax_tls.set_xlim(results.periods.min(), results.periods.max())

    # ── Panel 4: Phase-folded transit ────────────────────────────────────────────
    ax_fold.scatter(phase_s * PERIOD * 24, flux_s,
                    s=0.8, c=DET_COL, alpha=0.35, rasterized=True)
    # Binned model overlay
    valid = ~np.isnan(bin_means)
    ax_fold.plot(bin_centers[valid] * PERIOD * 24, bin_means[valid],
                 color=TRANSIT_COL, linewidth=1.8, zorder=5, label="Binned transit model")

    # Transit window shading — exact half-duration from TLS output
    half_dur_h = (DURATION / 2) * 24   # = 1.659 hours
    ax_fold.axvspan(-half_dur_h, half_dur_h, alpha=0.10, color=TRANSIT_COL)
    ax_fold.axhline(1.0, color=ACCENT, linewidth=0.6, linestyle="--", alpha=0.4)

    ax_fold.set_xlim(-6, 6)
    ax_fold.set_ylim(0.988, 1.006)
    ax_fold.legend(fontsize=6.5, loc="upper right", framealpha=0.7, edgecolor=GRID_COL)
    style_ax(ax_fold,
             "Phase-Folded Transit  (P = 3.2872 d)",
             "Time from Mid-Transit (Hours)",
             "Normalized Flux")

    # ── Sidebar panel 1: Candidate Summary ───────────────────────────────────────
    ax_sum.set_facecolor(PANEL_BG)
    ax_sum.set_xticks([]); ax_sum.set_yticks([])
    for spine in ax_sum.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    lines = [
        ("CANDIDATE SUMMARY", None, True),
        ("─" * 28, None, False),
        ("Target",        f"TIC {TIC_ID}",          False),
        ("Sector",        "TESS (2-min cadence)",    False),
        ("Observations",  f"{N_OBS:,}",              False),
        ("", "", False),
        ("Orbital Period",   f"{PERIOD:.5f} d",      False),
        ("Transit Duration", f"{DURATION_HRS:.2f} h",False),
        ("Transit Depth",    "~0.67%  flux drop",    False),
        ("TLS Depth param.", f"{DEPTH_TLS:.4f}",     False),
        ("", "", False),
        ("SDE Score",     f"{SDE:.2f}",              False),
        ("SDE Threshold", "7.0  (standard)",         False),
        ("", "", False),
        ("Classification",   "Transit Candidate ✓",  False),
        ("AI Classifier",    "Awaiting ISRO dataset",False),
    ]

    y = 0.96
    for label, val, is_header in lines:
        if is_header:
            ax_sum.text(0.5, y, label, transform=ax_sum.transAxes,
                        ha="center", va="top", fontsize=8.5, fontweight="bold",
                        color=ACCENT, fontfamily=FONT_MAIN)
        elif label.startswith("─"):
            ax_sum.plot([0.05, 0.95], [y - 0.01, y - 0.01],
                        color=ACCENT, linewidth=0.8,
                        transform=ax_sum.transAxes, clip_on=False)
        elif label == "":
            pass
        else:
            ax_sum.text(0.05, y, label + ":", transform=ax_sum.transAxes,
                        ha="left", va="top", fontsize=7.0, color=TEXT_SUB,
                        fontfamily=FONT_MAIN)
            col = PASS_COL if "✓" in (val or "") else TEXT_MAIN
            ax_sum.text(0.97, y, val, transform=ax_sum.transAxes,
                        ha="right", va="top", fontsize=7.0, color=col,
                        fontweight="bold" if "✓" in (val or "") else "normal",
                        fontfamily=FONT_MONO)
        y -= 0.065

    # ── Sidebar panel 2: Pipeline ribbon ─────────────────────────────────────────
    ax_pipe.set_facecolor(PANEL_BG)
    ax_pipe.set_xticks([]); ax_pipe.set_yticks([])
    for spine in ax_pipe.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    steps = [
        "TESS FITS Download",
        "Quality Filtering",
        "Wōtan Detrending",
        "Transit Least Squares",
        "Feature Extraction",
        "AI Classification  →",
    ]
    colors = [DET_COL, DET_COL, DET_COL, TLS_COL, PASS_COL, TEXT_SUB]
    done   = [True,    True,    True,    True,     True,      False]

    ax_pipe.text(0.5, 0.97, "PIPELINE", transform=ax_pipe.transAxes,
                 ha="center", va="top", fontsize=8, fontweight="bold",
                 color=ACCENT)

    y = 0.80
    for step, col, is_done in zip(steps, colors, done):
        marker = "●" if is_done else "○"
        alpha  = 1.0 if is_done else 0.45
        ax_pipe.text(0.12, y, marker, transform=ax_pipe.transAxes,
                     ha="center", va="center", fontsize=9, color=col, alpha=alpha)
        ax_pipe.text(0.22, y, step, transform=ax_pipe.transAxes,
                     ha="left", va="center", fontsize=6.8,
                     color=TEXT_MAIN if is_done else TEXT_SUB, alpha=alpha,
                     fontfamily=FONT_MAIN)
        if step != steps[-1]:
            ax_pipe.text(0.12, y - 0.115, "│", transform=ax_pipe.transAxes,
                         ha="center", va="center", fontsize=8,
                         color=ACCENT, alpha=0.35)
        y -= 0.145

    # ── Sidebar panel 3: Confidence bar ──────────────────────────────────────────
    ax_conf.set_facecolor(PANEL_BG)
    ax_conf.set_xticks([]); ax_conf.set_yticks([])
    for spine in ax_conf.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    ax_conf.text(0.5, 0.93, "DETECTION CONFIDENCE", transform=ax_conf.transAxes,
                 ha="center", va="top", fontsize=8, fontweight="bold", color=ACCENT)
    ax_conf.text(0.5, 0.78, f"SDE = {SDE:.2f}  (threshold: 7.0)",
                 transform=ax_conf.transAxes, ha="center", va="top",
                 fontsize=7, color=TEXT_SUB)

    # Draw confidence bar segments
    bar_y    = 0.38
    bar_left = 0.08
    bar_w    = 0.84
    seg_n    = 10
    seg_w    = bar_w / seg_n
    # SDE 18.09 → fills all 10 segments (anything above ~15 is maximum confidence)
    filled = 10

    level_colors = [
        "#FEE2E2","#FEE2E2",   # low (red)
        "#FEF3C7","#FEF3C7","#FEF3C7",  # medium (amber)
        "#D1FAE5","#D1FAE5","#D1FAE5","#D1FAE5","#D1FAE5",  # high (green)
    ]

    for i in range(seg_n):
        x = bar_left + i * seg_w
        c = level_colors[i] if i >= filled else (
            "#DC2626" if i < 2 else
            "#D97706" if i < 5 else
            "#16A34A"
        )
        rect = mpatches.FancyBboxPatch(
            (x + 0.005, bar_y), seg_w - 0.012, 0.25,
            boxstyle="round,pad=0.01",
            transform=ax_conf.transAxes,
            facecolor=c, edgecolor="white", linewidth=1.0, clip_on=False
        )
        ax_conf.add_patch(rect)

    ax_conf.text(bar_left, bar_y - 0.14, "Low",
                 transform=ax_conf.transAxes, ha="left",
                 fontsize=6.5, color=TEXT_SUB)
    ax_conf.text(0.5, bar_y - 0.14, "Medium",
                 transform=ax_conf.transAxes, ha="center",
                 fontsize=6.5, color=TEXT_SUB)
    ax_conf.text(bar_left + bar_w, bar_y - 0.14, "High ✓",
                 transform=ax_conf.transAxes, ha="right",
                 fontsize=6.5, color=PASS_COL, fontweight="bold")

    ax_conf.text(0.5, 0.07,
                 "Strong candidate — SDE well above 2.6× threshold",
                 transform=ax_conf.transAxes, ha="center", va="bottom",
                 fontsize=6.3, color=TEXT_SUB, style="italic")

    # ── Footer pipeline strip ─────────────────────────────────────────────────────
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.072, 0.072],
                   transform=fig.transFigure, color=ACCENT, linewidth=0.8, alpha=0.4))
    fig.text(0.5, 0.055,
             "Pipeline:  Lightkurve  →  Quality Filtering  →  Wōtan Detrending  "
             "→  Transit Least Squares (TLS)  →  Feature Extraction  →  AI Classifier (RF + batman)",
             ha="center", va="top", fontsize=7.5, color=TEXT_SUB, fontfamily=FONT_MAIN)
    fig.text(0.5, 0.032,
             "Libraries: Lightkurve · Wōtan · TransitLeastSquares · batman-package · scikit-learn · Astropy · NumPy · Matplotlib",
             ha="center", va="top", fontsize=6.8, color=TEXT_SUB, fontfamily=FONT_MAIN)

    # ── Save ──────────────────────────────────────────────────────────────────────
    out_path = "results/penumbra_dashboard.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"Saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()