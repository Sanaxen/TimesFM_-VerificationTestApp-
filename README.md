# TimesFM Time Series Forecasting App

[English](README.md)|[日本語](README_ja.md) 

A local time series forecasting app for Windows, built on Google Research's time series foundation model **TimesFM 2.5 (200M)**.
The UI is built with Streamlit, so you can go from loading a CSV to forecasting and viewing charts entirely in the browser.

## Requirements

- Windows 10 / 11
- Python 3.10 or later (on your PATH)
- Internet access (the model `google/timesfm-2.5-200m-pytorch` is downloaded automatically from Hugging Face the first time you run a forecast)

## Files

| File | Description |
| --- | --- |
| `setup.bat` | Creates the virtual environment (`venv`) and installs the libraries |
| `run.bat` | Starts the app (runs `setup.bat` first if `venv` does not exist) |
| `app.py` | The app itself |
| `requirements.txt` | List of required libraries |
| `sample.csv` | Sample data for testing (hourly, 2 columns) |
| `settings.json` | Created when you click "Save to app folder" |

## Setup and launch

1. Put the folder anywhere you like (outside `Downloads` is recommended; see Troubleshooting below).
2. Double-click `setup.bat`.
   - It creates `venv` and installs the contents of `requirements.txt`.
   - It then installs `timesfm[xreg]` (which uses JAX) for covariates as a separate step. If this fails you only get a warning; forecasting without covariates and the Ridge covariate method still work.
3. Double-click `run.bat`. The app opens in your browser.

## How to use

1. **Select a CSV.** The encoding (UTF-8 / Shift-JIS) and delimiter (comma, semicolon, tab) are detected automatically.
2. Choose the **date/time column** and the **columns to forecast** (multiple selection is supported).
3. Set the **validation period** (what percentage of the data, taken from the end, is used for validation) and the **forecast horizon** (number of rows to forecast beyond the end of the data).
4. If needed, set the **TimesFM parameters** and **covariates** (defaults are used at startup).
5. Click **Start forecast** to add the results to the chart.

### How validation and forecasting work

- **Validation forecast** (dotted line): uses only the data before the validation period to forecast the validation period, and reports the difference from the actual values as MAE / RMSE / MAPE.
- **Future forecast** (dashed line): uses all the data to forecast beyond the end.

These are two separate forecasts, so there can be a step between the end of the validation forecast and the last actual value. That step is the validation forecast error.

### Chart

- Each forecast column is drawn in its own color. The validation period is shaded orange and the future forecast period green.
- The view toggle switches between the full view and "Validation and forecast periods only".
- "Separate chart per column" shows columns with different scales in separate charts.
- "Show prediction interval (10-90%)" draws the forecast range as a band (not available with some covariate methods).
- The future forecast can be downloaded as a CSV.

## TimesFM parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `max_context` | 1024 | Maximum length of history used (multiple of 32, up to 16384) |
| `normalize_inputs` | On | Normalize the input |
| `use_continuous_quantile_head` | On | Continuous quantile head (required for prediction intervals) |
| `force_flip_invariance` | On | Enforce flip invariance |
| `infer_is_positive` | On | Prevent negative forecasts for positive-valued data |
| `fix_quantile_crossing` | On | Fix crossing quantiles |

## Covariates

### Available covariates

- **CSV columns**: numeric columns are treated as numbers, text columns as categories.
- **Generated from the date/time**: day of week, weekday/holiday (Japanese public holidays can be included), AM/PM, season (4 seasons), and month (1-12). Features with a constant value (for example AM/PM in daily data) are excluded automatically.

### When CSV covariate values are missing for the forecast period

Using a CSV column as a covariate requires its values for the forecast period.

- If you add rows at the end of the CSV where the forecast column is empty and only the covariate has values, those values are used.
- Any shortfall is filled in one of two ways:
  - **Fill with the last value** (default)
  - **Forecast with TimesFM and fill** (numeric columns only; text columns use the last value). If the covariate forecast is off, the error carries over into the results.

### Covariate methods

| Method | Description |
| --- | --- |
| TimesFM XReg (requires JAX) | TimesFM's built-in covariate feature. `xreg_mode` and `ridge` can be set. |
| Ridge regression + TimesFM (no JAX) | Removes the effect of the covariates with Ridge regression, forecasts the residual with TimesFM, then adds the effect back. Handles linear relationships only. |

In environments where JAX cannot be loaded, the default switches to the Ridge method automatically.

Validation uses the actual CSV values for the covariates, so the validation results show accuracy when the covariates are known correctly.

## Saving and loading settings

The "Save / load settings" section in the sidebar saves and loads all settings except the CSV (column selections, periods, parameters, and covariates).

- "Download settings (JSON)": saves the settings as a JSON file.
- "Save to app folder (settings.json)" / "Load settings.json": saves and loads in the app folder.
- Choose a settings file and click "Apply selected settings": applies a saved JSON file.

When applied to a different CSV, columns that do not exist are ignored and out-of-range values are clamped. Chart display options are not saved.

## Troubleshooting

### `DLL load failed ... blocked by an application control policy`

Windows application control (Smart App Control / WDAC / AppLocker) is blocking a Python library DLL from loading. Try the following.

- Run `run.bat` **as administrator**.
- Move the folder outside `Downloads` (for example `C:\dev\timesfm_app`), delete `venv`, and run `setup.bat` again.
- If Smart App Control is the cause, check the setting in Windows Security (once turned off, it cannot be turned back on without reinstalling Windows).
- On a work or school PC managed by an administrator, ask your administrator.

### `Failed to load the XReg module`

JAX could not be loaded. Switch the covariate method to "Ridge regression + TimesFM (no JAX)", or investigate the cause:

```
venv\Scripts\activate
pip install "timesfm[xreg]"
python -c "import jax"
```

### Cannot select columns to forecast

Check that a numeric column has not been selected as the date/time column. You can also open "Data preview" to confirm that the columns were split correctly.

### The first forecast is slow

The model is being downloaded and loaded. Later runs are faster. Switching covariates on or off reloads the model.

### Using a GPU

The PyTorch installed by `pip` is the CPU build. To use a GPU, reinstall the CUDA build of PyTorch into `venv` by following the official PyTorch instructions.

## License and credits

- TimesFM: [google-research/timesfm](https://github.com/google-research/timesfm) (Apache-2.0)
- This app uses the official TimesFM package (`timesfm` on PyPI).
