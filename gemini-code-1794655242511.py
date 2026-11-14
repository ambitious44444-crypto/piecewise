import io
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import streamlit as st

st.set_page_config(page_title="Piecewise Linear Regression Analyzer", layout="wide")

st.title("📈 Piecewise Linear Regression & Breakpoint Analyzer")
st.markdown(
    "Upload multiple `(x, y)` CSV files, specify the number of theoretical breakpoints, "
    "and extract fitted slopes, breakpoint $x$-values, and publication-ready figures."
)

# --- 1. SIDEBAR CONTROLS: BREAKPOINT DESIGNATION ---
st.sidebar.header("Model Configuration")
n_breakpoints = st.sidebar.number_input(
    "Theoretical Number of Breakpoints",
    min_value=1,
    max_value=5,
    value=1,
    step=1,
    help="Number of slope changes (kinks) to fit in the continuous piecewise linear model.",
)
has_header = st.sidebar.checkbox("CSV files contain a header row", value=False)


# --- PIECEWISE LINEAR MODEL & OPTIMIZER ---
def piecewise_linear(x, intercept, slope0, delta_slopes, breakpoints):
    """Continuous piecewise linear function using ReLU basis functions."""
    y = intercept + slope0 * x
    for dm, bp in zip(delta_slopes, breakpoints):
        y += dm * np.maximum(0.0, x - bp)
    return y


def fit_piecewise(x, y, k):
    """Fits k breakpoints using bounded non-linear least squares."""
    x_min, x_max = np.min(x), np.max(x)
    x_span = x_max - x_min
    margin = 0.05 * x_span  # Prevent breakpoints from collapsing onto endpoints

    # Initial guesses: evenly spaced breakpoints, initial linear trend
    bp_init = np.linspace(x_min + 0.2 * x_span, x_max - 0.2 * x_span, k)
    slope_init = (y[-1] - y[0]) / x_span if x_span != 0 else 0.0
    intercept_init = y[0] - slope_init * x[0]
    delta_slopes_init = np.zeros(k)

    # Parameter vector: [intercept, slope0, dm_1...dm_k, bp_1...bp_k]
    p0 = np.concatenate([[intercept_init, slope_init], delta_slopes_init, bp_init])

    # Bounds: unbounded slopes/intercepts; breakpoints restricted within data range
    lower_bounds = np.concatenate([
        [-np.inf, -np.inf],
        np.full(k, -np.inf),
        np.full(k, x_min + margin),
    ])
    upper_bounds = np.concatenate([
        [np.inf, np.inf],
        np.full(k, np.inf),
        np.full(k, x_max - margin),
    ])

    def residuals(params):
        intercept = params[0]
        slope0 = params[1]
        delta_slopes = params[2 : 2 + k]
        bps = params[2 + k :]
        return piecewise_linear(x, intercept, slope0, delta_slopes, bps) - y

    res = least_squares(residuals, p0, bounds=(lower_bounds, upper_bounds), max_nfev=20000)

    # Extract fitted parameters and sort breakpoints in ascending order
    popt = res.x
    intercept = popt[0]
    slope0 = popt[1]
    delta_slopes = popt[2 : 2 + k]
    bps = popt[2 + k :]

    # Compute standard errors from Jacobian covariance approximation
    dof = max(1, len(x) - len(popt))
    rss = np.sum(res.fun**2)
    s_sq = rss / dof
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * s_sq
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        perr = np.full(len(popt), np.nan)

    bp_errs = perr[2 + k :]

    # Sort breakpoints chronologically/spatially along x
    sort_idx = np.argsort(bps)
    bps_sorted = bps[sort_idx]
    bp_errs_sorted = bp_errs[sort_idx]

    # Recompute segment slopes cleanly in sorted order
    sorted_delta_slopes = delta_slopes[sort_idx]
    segment_slopes = [slope0]
    for dm in sorted_delta_slopes:
        segment_slopes.append(segment_slopes[-1] + dm)

    # R-squared calculation
    tss = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (rss / tss) if tss > 0 else np.nan

    y_fit = piecewise_linear(x, intercept, slope0, sorted_delta_slopes, bps_sorted)
    return {
        "intercept": intercept,
        "slope0": slope0,
        "delta_slopes": sorted_delta_slopes,
        "segment_slopes": segment_slopes,
        "breakpoints": bps_sorted,
        "bp_errors": bp_errs_sorted,
        "r_squared": r_squared,
        "y_fit": y_fit,
        "residuals": y - y_fit,
    }


# --- 2. FILE UPLOAD ---
uploaded_files = st.file_uploader(
    "Upload CSV files containing (x, y) columns",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda f: f.name)
    st.success(f"Loaded {len(uploaded_files)} file(s). Running piecewise regression with {n_breakpoints} breakpoint(s)...")

    summary_rows = []
    figure_buffers = {}

    # Layout grid for figures (2 columns)
    cols = st.columns(2)

    for idx, file in enumerate(uploaded_files):
        file.seek(0)
        header_arg = 0 if has_header else None
        df = pd.read_csv(file, header=header_arg)

        # Enforce numeric conversion and sort by x
        x_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        y_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        valid = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        x, y = x_raw[valid], y_raw[valid]

        sort_x = np.argsort(x)
        x, y = x[sort_x], y[sort_x]

        if len(x) < (2 * n_breakpoints + 2):
            st.error(f"**{file.name}**: Not enough data points ({len(x)}) to fit {n_breakpoints} breakpoint(s).")
            continue

        # --- 3. RUN PIECEWISE LINEAR REGRESSION ---
        fit = fit_piecewise(x, y, n_breakpoints)

        # --- 4. GENERATE & PRINT FITTING FIGURES ---
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6.5, 5.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # Main fit subplot
        ax1.scatter(x, y, color="#2b5c8f", alpha=0.75, s=28, label="Experimental Data")
        x_dense = np.linspace(np.min(x), np.max(x), 400)
        y_dense = piecewise_linear(
            x_dense, fit["intercept"], fit["slope0"], fit["delta_slopes"], fit["breakpoints"]
        )
        ax1.plot(x_dense, y_dense, color="#d9534f", linewidth=2.2, label=f"Piecewise Fit ($R^2$ = {fit['r_squared']:.4f})")

        # Annotate breakpoints
        for b_idx, (bp, bp_err) in enumerate(zip(fit["breakpoints"], fit["bp_errors"]), start=1):
            ax1.axvline(bp, color="#2ca02c", linestyle="--", linewidth=1.4, label=f"BP {b_idx}: x = {bp:.2f}")
            ax1.text(
                bp,
                ax1.get_ylim()[1] * 0.95,
                f" BP{b_idx}={bp:.2f}",
                color="#1e701e",
                fontsize=9,
                fontweight="bold",
                verticalalignment="top",
            )

        ax1.set_title(f"{file.name}", fontsize=11, fontweight="bold")
        ax1.set_ylabel("y", fontsize=10)
        ax1.legend(loc="best", fontsize=8.5)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Residuals subplot
        ax2.axhline(0, color="black", linestyle="-", linewidth=1)
        ax2.scatter(x, fit["residuals"], color="#555555", s=18, alpha=0.8)
        ax2.set_ylabel("Residuals", fontsize=9)
        ax2.set_xlabel("x", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()

        # Render figure in Streamlit grid
        with cols[idx % 2]:
            st.pyplot(fig)

        # Save figure into memory buffer for export
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        base_name = file.name.rsplit(".", 1)[0]
        figure_buffers[f"{base_name}_piecewise_fit.png"] = buf.getvalue()
        plt.close(fig)

        # --- 5. BUILD SUMMARY TABLE ROW ---
        row = {
            "Figure / File": file.name,
            "R_squared": round(fit["r_squared"], 5),
        }
        for b_idx, (bp, bp_err) in enumerate(zip(fit["breakpoints"], fit["bp_errors"]), start=1):
            row[f"Breakpoint_{b_idx}_x"] = round(bp, 4)
            row[f"Breakpoint_{b_idx}_SE"] = round(bp_err, 4) if not np.isnan(bp_err) else np.nan

        for s_idx, slope in enumerate(fit["segment_slopes"], start=1):
            row[f"Slope_Segment_{s_idx}"] = round(slope, 5)

        summary_rows.append(row)

    # --- DISPLAY TABLE & DOWNLOAD BUTTONS (STEPS 5 & 6) ---
    if summary_rows:
        st.divider()
        st.subheader("📊 Extracted Breakpoint $x$-Values & Slopes")
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)

        # Save table as CSV
        with col_dl1:
            csv_bytes = df_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Breakpoint Table (.csv)",
                data=csv_bytes,
                file_name="piecewise_breakpoints_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Save all figures as a single ZIP archive
        with col_dl2:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, img_bytes in figure_buffers.items():
                    zf.writestr(fname, img_bytes)
            zip_buf.seek(0)

            st.download_button(
                label="🖼️ Download All Figures (.zip, 300 DPI PNG)",
                data=zip_buf,
                file_name="piecewise_regression_figures.zip",
                mime="application/zip",
                use_container_width=True,
            )