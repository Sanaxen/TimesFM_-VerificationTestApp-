import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

st.set_page_config(page_title="TimesFM Forecast", layout="wide")
COLORS = qualitative.Plotly


# ---------------------------------------------------------------- i18n
TR = {
    "title": ("TimesFM 時系列予測アプリ", "TimesFM Time Series Forecasting App"),
    "csv_enc_err": ("CSVの文字コードを判別できませんでした（UTF-8 / Shift-JIS に対応）",
                    "Could not detect the CSV encoding (UTF-8 / Shift-JIS supported)"),
    "model_loading": ("TimesFMモデルを読み込み中（初回はダウンロードのため時間がかかります）...",
                      "Loading the TimesFM model (the first run downloads it and takes a while)..."),
    "actual": ("実測", "Actual"),
    "val_fc": ("検証予測", "Validation forecast"),
    "fut_fc": ("未来予測", "Future forecast"),
    "cal:dow": ("曜日", "Day of week"),
    "cal:holiday": ("平日/休日", "Weekday/Holiday"),
    "cal:ampm": ("午前/午後", "AM/PM"),
    "cal:season": ("季節", "Season"),
    "cal:month": ("月", "Month"),
    "engine_xreg": ("TimesFM XReg（JAX・scikit-learn必須）", "TimesFM XReg (requires JAX and scikit-learn)"),
    "engine_ridge": ("Ridge回帰＋TimesFM（JAX不要）", "Ridge regression + TimesFM (no JAX)"),
    "fill_last": ("最後の値で埋める", "Fill with the last value"),
    "fill_forecast": ("TimesFMで予測して埋める", "Forecast with TimesFM and fill"),
    "upload_csv": ("CSVファイルを選択", "Select a CSV file"),
    "upload_info": ("CSVファイルをアップロードしてください。", "Please upload a CSV file."),
    "sb_header": ("設定の保存 / 読み込み", "Save / load settings"),
    "sb_caption": ("CSV以外の設定（列の選択・期間・パラメータ・説明変数）が対象です。",
                   "Covers all settings except the CSV (column selections, periods, parameters, covariates)."),
    "dl_settings": ("設定をダウンロード (JSON)", "Download settings (JSON)"),
    "save_local": ("アプリのフォルダに保存 (settings.json)", "Save to app folder (settings.json)"),
    "saved": ("保存しました: {p}", "Saved: {p}"),
    "save_fail": ("保存に失敗しました: {e}", "Failed to save: {e}"),
    "pick_cfg": ("設定ファイル (JSON) を選択", "Select a settings file (JSON)"),
    "apply_sel": ("選択した設定を適用", "Apply selected settings"),
    "load_local": ("settings.json を読込", "Load settings.json"),
    "applied": ("設定を適用しました。", "Settings applied."),
    "cfg_fail": ("設定ファイルを読み込めませんでした: {e}", "Could not read the settings file: {e}"),
    "preview": ("データプレビュー", "Data preview"),
    "col_time": ("日時・時間の項目", "Date/time column"),
    "no_num": ("数値として読める項目が見つかりません（読み込んだ列: {cols}）。日時項目の選択、または区切り文字・ヘッダー行を確認してください。",
               "No column could be read as numeric (columns found: {cols}). Check the date/time column selection, the delimiter, and the header row."),
    "series_col": ("系列名の列（ロング形式のデータ）", "Series column (long-format data)"),
    "series_none": ("（なし：列ごとに系列）", "(none: each column is a series)"),
    "series_help": ("「日時・系列名・値…」のように、1つの日時に系列ごとの行が並ぶ形式のときに、系列名が入っている列を選びます。"
                    "系列ごと・予測項目ごとに、別々の時系列として予測します。",
                    "Choose the column holding the series names when the data has one row per series for each timestamp "
                    "(e.g. date, name, values...). Each series and each forecast column is forecast as a separate time series."),
    "series_sel": ("予測する系列", "Series to forecast"),
    "long_info": ("{k}系列 × {m}項目 = {s}本の時系列を予測します。", "Forecasting {s} time series ({k} series x {m} columns)."),
    "pick_series": ("予測する系列を1つ以上選択してください。", "Please select at least one series."),
    "dup_dates": ("日時が重複しています。1つの日時に複数行ある場合は、「系列名の列」を選択してください。",
                  "The date/time column has duplicates. If there are several rows per timestamp, choose a series column."),
    "no_valid": ("有効なデータがないため除外した系列: {names}", "Series excluded because they have no valid data: {names}"),
    "val_skipped": ("検証開始時点で学習に使えるデータが32行未満のため、検証を行わない系列: {names}",
                    "Series skipped in validation because fewer than 32 rows precede the validation start: {names}"),
    "col_targets": ("予測するデータ項目（複数選択可）", "Columns to forecast (multiple selection allowed)"),
    "val_pct": ("検証期間（データ全体に対する後半の割合 %）", "Validation period (% of the data, taken from the end)"),
    "val_pct_help": ("0にすると検証を行わず、未来予測のみ実行します", "Set to 0 to skip validation and run only the future forecast"),
    "horizon": ("予測期間（データ末尾から先の行数）", "Forecast horizon (rows beyond the end of the data)"),
    "from_val": ("検証区間の開始点から予測を開始する", "Start the forecast from the beginning of the validation period"),
    "from_val_help": ("オンにすると、検証期間より前のデータだけを使い、検証期間の先頭から「検証期間＋予測期間」ぶんを1回で予測します。"
                      "検証予測と未来予測が途切れずにつながりますが、検証期間の実測値は予測に使われません。検証期間が0%のときは無効です。",
                      "When on, only the data before the validation period is used, and one forecast covers 'validation period + horizon' "
                      "from the start of the validation period. The validation and future forecasts connect without a gap, but the actual "
                      "values in the validation period are not used for forecasting. Disabled when the validation period is 0%."),
    "cont_note": ("検証区間の開始点から、検証期間＋予測期間を1回で予測しています。",
                  "One forecast covers the validation period plus the horizon, starting at the beginning of the validation period."),
    "params_exp": ("TimesFM 予測パラメータ（初期値はデフォルト）", "TimesFM parameters (defaults at startup)"),
    "max_context": ("max_context（参照する過去の最大長, 32の倍数）", "max_context (maximum history length, multiple of 32)"),
    "max_context_help": ("長いほど多くの履歴を使う。TimesFM 2.5は最大16k", "Longer values use more history. TimesFM 2.5 supports up to 16k"),
    "normalize": ("normalize_inputs（入力の正規化）", "normalize_inputs (normalize the input)"),
    "qhead": ("use_continuous_quantile_head（連続分位点ヘッド）", "use_continuous_quantile_head (continuous quantile head)"),
    "flip": ("force_flip_invariance（反転不変性）", "force_flip_invariance (flip invariance)"),
    "positive": ("infer_is_positive（正値データなら負値予測を防ぐ）", "infer_is_positive (avoid negative forecasts for positive data)"),
    "fixq": ("fix_quantile_crossing（分位点の交差補正）", "fix_quantile_crossing (fix crossing quantiles)"),
    "cov_exp": ("説明変数（共変量）", "Covariates"),
    "csv_covs": ("CSVから選択（数値列は数値、文字列列はカテゴリとして扱う）",
                 "Select from the CSV (numeric columns as numbers, text columns as categories)"),
    "csv_covs_help": ("予測期間の値が必要です。CSV末尾に『予測項目が空欄で説明変数だけ入っている行』があればそれを使い、足りない分は下の設定に従って埋めます。",
                      "Values for the forecast period are required. Rows at the end of the CSV where the forecast column is empty but the covariate has values are used; any shortfall is filled according to the setting below."),
    "cal_opts": ("日時から自動追加", "Add automatically from the date/time"),
    "jp_hol": ("『休日』に日本の祝日を含める（jpholiday）", "Include Japanese public holidays in 'holiday' (jpholiday)"),
    "fill_mode": ("CSVの説明変数が予測期間で不足する場合（数値列）", "When CSV covariates are missing for the forecast period (numeric columns)"),
    "fill_help": ("『TimesFMで予測して埋める』は、その説明変数自体を単独で予測して不足分に使います。文字列列は最後の値で埋めます。",
                  "'Forecast with TimesFM and fill' forecasts the covariate itself and uses it for the shortfall. Text columns are filled with the last value."),
    "engine": ("共変量の方式", "Covariate method"),
    "engine_help": ("Ridge回帰＋TimesFM: 説明変数の効果を回帰で取り除いた残差をTimesFMで予測し、効果を足し戻します。",
                    "Ridge regression + TimesFM: removes the covariates' effect by regression, forecasts the residual with TimesFM, then adds the effect back."),
    "jax_caption": ("※ JAX または scikit-learn を読み込めないため、初期値をJAX不要の方式にしています（XReg方式を選んでも、自動でRidge方式に切り替えます）。",
                    "Note: JAX or scikit-learn could not be loaded, so the default is the method that does not need them (choosing XReg falls back to Ridge automatically)."),
    "xreg_fallback": ("XReg方式に必要な JAX / scikit-learn を読み込めないため、Ridge回帰＋TimesFM方式で予測します。"
                      "XReg方式を使うには、仮想環境で `pip install jax scikit-learn` を実行してください。",
                      "JAX / scikit-learn (needed for the XReg method) could not be loaded, so the Ridge regression + TimesFM method is used. "
                      "To use XReg, run `pip install jax scikit-learn` in the virtual environment."),
    "xmode": ("xreg_mode（XReg方式のみ）", "xreg_mode (XReg method only)"),
    "ridge": ("ridge（正則化）", "ridge (regularization)"),
    "pick_target": ("予測するデータ項目を1つ以上選択してください。", "Please select at least one column to forecast."),
    "few_rows": ("有効なデータ行が少なすぎます（40行以上を推奨）。日時列の選択を確認してください。",
                 "Too few valid data rows (40 or more recommended). Check the date/time column selection."),
    "start": ("予測開始", "Start forecast"),
    "val_too_long": ("検証期間が長すぎて学習に使える行数が不足しています。割合を下げてください。",
                     "The validation period is too long and leaves too few rows for training. Lower the percentage."),
    "warn_fill_forecast": ("CSVに未来の説明変数が{nf}行分しかないため、不足分はTimesFMで予測して埋めています（文字列列は最後の値）。",
                           "The CSV has covariate values for only {nf} future row(s); the shortfall is forecast with TimesFM and filled (text columns use the last value)."),
    "warn_fill_last": ("CSVに未来の説明変数が{nf}行分しかないため、不足分は最後の値で埋めています。",
                       "The CSV has covariate values for only {nf} future row(s); the shortfall is filled with the last value."),
    "dropped": ("値が一定で意味を持たないため除外した説明変数: {names}", "Covariates excluded because their values are constant: {names}"),
    "predicting": ("予測中...", "Forecasting..."),
    "fail": ("予測に失敗しました: {e}", "Forecast failed: {e}"),
    "col_item": ("項目", "Column"),
    "graph": ("グラフ", "Chart"),
    "view": ("表示切り替え", "View"),
    "view_full": ("全体表示", "Full view"),
    "view_only": ("検証・予測区間のみ", "Validation and forecast periods only"),
    "split": ("項目ごとにグラフを分ける", "Separate chart per column"),
    "band": ("予測区間(10-90%)を表示", "Show prediction interval (10-90%)"),
    "cov_used": ("使用した説明変数: {names}", "Covariates used: {names}"),
    "val_result": ("検証結果", "Validation results"),
    "dl_forecast": ("予測結果をCSVでダウンロード", "Download the forecast as CSV"),
}
LEGACY = {  # 旧バージョンの設定ファイル（日本語の値）を内部IDへ変換
    "TimesFM XReg（JAX必須）": "xreg", "Ridge回帰＋TimesFM（JAX不要）": "ridge",
    "最後の値で埋める": "last", "TimesFMで予測して埋める": "forecast",
    "曜日": "dow", "平日/休日": "holiday", "午前/午後": "ampm", "季節": "season", "月": "month",
}


def t(key, **kw):
    s = TR[key][0 if st.session_state.get("lang", "ja") == "ja" else 1]
    return s.format(**kw) if kw else s


st.session_state.setdefault("lang", "ja")
st.sidebar.radio("Language / 言語", ["ja", "en"], format_func={"ja": "日本語", "en": "English"}.get,
                 horizontal=True, key="lang")
st.title(t("title"))



# ---------------------------------------------------------------- helpers
@st.cache_data
def load_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise ValueError(t("csv_enc_err"))


@st.cache_resource(max_entries=1, show_spinner=False)
def get_model(max_context, max_horizon, normalize, qhead, flip, positive, fixq, backcast):
    import timesfm
    import torch

    torch.set_float32_matmul_precision("high")
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=normalize,
            use_continuous_quantile_head=qhead,
            force_flip_invariance=flip,
            infer_is_positive=positive,
            fix_quantile_crossing=fixq,
            return_backcast=backcast,
        )
    )
    return m


def to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str).str.replace(r"[,\s%]", "", regex=True), errors="coerce")


def guess_time_col(df: pd.DataFrame) -> int:
    for i, c in enumerate(df.columns):
        if df[c].dtype == object:
            ok = pd.to_datetime(df[c].head(100), errors="coerce").notna().mean()
            if ok > 0.9:
                return i
    return 0


def calendar_feats(ts, opts, use_jp_holiday):
    ts = pd.DatetimeIndex(ts)
    f = {}
    if "dow" in opts:
        f["cal:dow"] = np.asarray(ts.dayofweek)
    if "holiday" in opts:
        off = np.asarray(ts.dayofweek >= 5)
        if use_jp_holiday:
            try:
                import jpholiday

                off = off | np.array([jpholiday.is_holiday(x.date()) for x in ts], dtype=bool)
            except ImportError:
                pass
        f["cal:holiday"] = off.astype(int)
    if "ampm" in opts:
        f["cal:ampm"] = (np.asarray(ts.hour) >= 12).astype(int)
    if "season" in opts:
        f["cal:season"] = (np.asarray(ts.month) % 12) // 3  # 0:冬(12-2) 1:春 2:夏 3:秋
    if "month" in opts:
        f["cal:month"] = np.asarray(ts.month)
    return f


def build_design(cov_num, cov_cat, start, end, H):
    """1系列分の共変量から回帰用の説明行列を作る（数値は標準化、カテゴリはone-hot）。行数は (end-start)+H。"""
    L = end - start
    cols_ = []
    for v in cov_num.values():
        x = np.asarray(v[start : end + H], dtype=float)
        sd = x[:L].std()
        cols_.append((x - x[:L].mean()) / (sd if sd > 0 else 1.0))
    for v in cov_cat.values():
        x = np.asarray(v[start : end + H])
        for cat in np.unique(x[:L])[1:]:  # 先頭カテゴリを基準にして多重共線性を避ける
            cols_.append((x == cat).astype(float))
    return np.column_stack(cols_) if cols_ else None


def ridge_forecast(model, inputs, Xs, H, ridge):
    """JAX不要の代替: Ridge回帰で説明変数の効果を除き、残差をTimesFMで予測して効果を足し戻す。"""
    resid, shifts = [], []
    for y, X in zip(inputs, Xs):
        y = np.asarray(y, dtype=float)
        L = len(y)
        Xc, Xf = X[:L], X[L:]
        A = np.column_stack([np.ones(L), Xc])
        reg = max(float(ridge), 1e-3) * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(A.T @ A + reg, A.T @ y)
        resid.append(y - Xc @ w[1:])  # 切片は残すのでレベルは保たれる
        shifts.append(Xf @ w[1:])
    p, q = model.forecast(horizon=H, inputs=resid)
    shifts = np.array(shifts)
    p = np.asarray(p) + shifts
    q = np.asarray(q) + shifts[:, :, None] if q is not None else None
    return p, q


def run_forecast(model, arrs, end, H, cov_num, cov_cat, max_context, xmode, ridge, use_xreg=True, los=None):
    """各系列の arrs[i][:end] を入力に H 先を予測。
    cov_num / cov_cat は {名前: [系列ごとの配列(履歴＋未来)]}。los は各系列の有効データ開始位置。"""
    los = los or [0] * len(arrs)
    starts = [max(0, end - int(max_context), lo) for lo in los]
    inputs = [a[st:end] for a, st in zip(arrs, starts)]
    if not cov_num and not cov_cat:
        p, q = model.forecast(horizon=H, inputs=inputs)
    elif not use_xreg:
        Xs = [build_design({k: v[i] for k, v in cov_num.items()}, {k: v[i] for k, v in cov_cat.items()}, st, end, H)
              for i, st in enumerate(starts)]
        p, q = ridge_forecast(model, inputs, Xs, H, ridge)
    else:
        kw = {}
        if cov_num:
            kw["dynamic_numerical_covariates"] = {
                k: [np.asarray(v[i][st : end + H], dtype=float).tolist() for i, st in enumerate(starts)]
                for k, v in cov_num.items()}
        if cov_cat:
            kw["dynamic_categorical_covariates"] = {
                k: [np.asarray(v[i][st : end + H]).tolist() for i, st in enumerate(starts)]
                for k, v in cov_cat.items()}
        p, q = model.forecast_with_covariates(inputs=inputs, xreg_mode=xmode, ridge=ridge, **kw)
    q = np.asarray(q) if q is not None else None
    if q is not None and (q.ndim != 3 or q.shape[-1] < 10):
        q = None
    return np.asarray(p), q


def future_index(ts: pd.Series, h: int) -> pd.DatetimeIndex:
    try:
        freq = pd.infer_freq(pd.DatetimeIndex(ts))
    except Exception:
        freq = None
    if freq:
        return pd.date_range(ts.iloc[-1], periods=h + 1, freq=freq)[1:]
    delta = ts.diff().median()
    return pd.DatetimeIndex([ts.iloc[-1] + delta * (i + 1) for i in range(h)])


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def make_fig(cols, targets, d, tcol, res, only_forecast, show_band):
    fig = go.Figure()
    n = len(d)
    v = res["v"] if res else 0
    start = (n - v if v > 0 else n - 1) if (res and only_forecast) else 0
    x_act = d[tcol].iloc[start:]
    for c in cols:
        color = COLORS[targets.index(c) % len(COLORS)]
        fig.add_trace(go.Scatter(x=x_act, y=d[c].iloc[start:], name=f"{c} ({t('actual')})", line=dict(color=color)))
        if not res:
            continue
        k = res["targets"].index(c)
        if v > 0 and not np.all(np.isnan(res["val_pred"][k])):
            xv = d[tcol].iloc[n - v :]
            vq = res.get("val_q")
            if show_band and vq is not None and np.isfinite(vq[k]).all():
                fig.add_trace(go.Scatter(x=xv, y=vq[k][:, 9], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=xv, y=vq[k][:, 1], line=dict(width=0), fill="tonexty",
                                         fillcolor=rgba(color, 0.2), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xv, y=res["val_pred"][k], name=f"{c} ({t('val_fc')})",
                                     line=dict(color=color, dash="dot", width=2)))
        xf = [d[tcol].iloc[-1]] + list(res["fut_idx"])
        y0 = res["val_pred"][k][-1] if (res.get("cont") and v > 0 and np.isfinite(res["val_pred"][k][-1])) else d[c].iloc[-1]
        yf = [y0] + list(res["fut_pred"][k])
        if show_band and res["fut_q"] is not None:
            lo, hi = res["fut_q"][k][:, 1], res["fut_q"][k][:, 9]
            fig.add_trace(go.Scatter(x=list(res["fut_idx"]), y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=list(res["fut_idx"]), y=lo, line=dict(width=0), fill="tonexty",
                                     fillcolor=rgba(color, 0.2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xf, y=yf, name=f"{c} ({t('fut_fc')})", line=dict(color=color, dash="dash", width=2)))
    if res:
        if v > 0:
            fig.add_vrect(x0=d[tcol].iloc[n - v].to_pydatetime(), x1=d[tcol].iloc[-1].to_pydatetime(),
                          fillcolor="orange", opacity=0.08, line_width=0, layer="below")
        fig.add_vrect(x0=d[tcol].iloc[-1].to_pydatetime(), x1=res["fut_idx"][-1].to_pydatetime(),
                      fillcolor="green", opacity=0.08, line_width=0, layer="below")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15))
    return fig


# ---------------------------------------------------------------- input
SETTINGS_PATH = Path(__file__).with_name("settings.json")
ENGINES = ["xreg", "ridge"]
FILL_MODES = ["last", "forecast"]
XMODES = ["xreg + timesfm", "timesfm + xreg"]
CAL_ALL = ["dow", "holiday", "ampm", "season", "month"]
# 保存・読み込み対象の設定（CSV以外）: key -> (型, 既定値, 範囲/選択肢)
SPEC = {
    "val_pct": ("int", 20, (0, 50)),
    "horizon": ("int", 24, (1, 16000)),
    "from_val": ("bool", False, None),
    "max_context": ("int", 1024, (32, 16384)),
    "normalize": ("bool", True, None),
    "qhead": ("bool", True, None),
    "flip": ("bool", True, None),
    "positive": ("bool", True, None),
    "fixq": ("bool", True, None),
    "cal_opts": ("multi", [], CAL_ALL),
    "jp_hol": ("bool", True, None),
    "fill_mode": ("choice", FILL_MODES[0], FILL_MODES),
    "engine": ("choice", ENGINES[0], ENGINES),
    "xmode": ("choice", XMODES[0], XMODES),
    "ridge": ("float", 0.0, (0.0, 1000.0)),
}
CFG_KEYS = ["tcol", "series_col", "series_sel", "targets", "csv_covs"] + list(SPEC)


def coerce(key, v):
    typ, _, ext = SPEC[key]
    try:
        if typ == "bool":
            return v if isinstance(v, bool) else None
        if typ in ("int", "float"):
            v = min(max(float(v), ext[0]), ext[1])
            if typ == "int":
                v = int(v)
                if key == "max_context":
                    v = max(32, v // 32 * 32)
            return v
        if typ == "choice":
            v = LEGACY.get(v, v) if isinstance(v, str) else v
            return v if v in ext else None
        if typ == "multi":
            if not isinstance(v, list):
                return None
            v = [LEGACY.get(x, x) if isinstance(x, str) else x for x in v]
            return [x for x in v if x in ext]
    except (TypeError, ValueError):
        return None


up = st.file_uploader(t("upload_csv"), type=["csv"])
if up is None:
    st.info(t("upload_info"))
    st.stop()

df = load_csv(up.getvalue())
cols = list(df.columns)
ss = st.session_state

try:
    import jax  # noqa: F401
    from sklearn import preprocessing  # noqa: F401  (TimesFMのXRegが必要とする)

    jax_ok = True
except Exception:  # noqa: BLE001
    jax_ok = False

for k, (_, dflt, _) in SPEC.items():
    ss.setdefault(k, list(dflt) if isinstance(dflt, list) else dflt)
if "engine_init" not in ss:
    ss["engine"] = ENGINES[0] if jax_ok else ENGINES[1]
    ss["engine_init"] = True


def detect_series_col(tc):
    """同じ日時に複数行あり、(日時, 列) が一意になる文字列列を系列名の列とみなす。"""
    if not df[tc].duplicated().any():
        return ""
    for c in cols:
        if c != tc and not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 200 \
                and not df.duplicated([tc, c]).any():
            return c
    return ""


def series_values(sc):
    return sorted(df[sc].dropna().astype(str).unique()) if sc else []


def sync_state():
    """列に依存する選択（日時・系列・予測項目・説明変数）を、現在のCSVで有効な値に整える。"""
    if ss.get("tcol") not in cols:
        ss["tcol"] = cols[guess_time_col(df)]
    sc = ss.get("series_col")
    if sc is None or (sc and sc not in cols):
        sc = detect_series_col(ss["tcol"])
    ss["series_col"] = "" if sc == ss["tcol"] else sc
    sc = ss["series_col"]
    vals = series_values(sc)
    cur_s = ss.get("series_sel")
    valid_s = [x for x in (cur_s or []) if x in vals]
    ss["series_sel"] = (vals[:10] if (cur_s is None or (cur_s and not valid_s) or not cur_s and sc and "_sel_init" not in ss)
                        else valid_s) if sc else []
    if sc:
        ss["_sel_init"] = True
    nc = [c for c in cols if c not in (ss["tcol"], sc) and to_num(df[c]).notna().mean() > 0.5]
    cur = ss.get("targets")
    valid = [c for c in (cur or []) if c in nc]
    ss["targets"] = nc[:1] if (cur is None or (cur and not valid)) else valid
    ss["csv_covs"] = [c for c in ss.get("csv_covs", []) if c in cols and c not in (ss["tcol"], sc) and c not in ss["targets"]]
    return nc


def apply_cfg(cfg):
    if cfg.get("tcol") in cols:
        ss["tcol"] = cfg["tcol"]
    if cfg.get("series_col") in cols or cfg.get("series_col") == "":
        ss["series_col"] = cfg["series_col"]
    if isinstance(cfg.get("series_sel"), list):
        ss["series_sel"] = [str(x) for x in cfg["series_sel"]]
    for k in SPEC:
        if k in cfg:
            v = coerce(k, cfg[k])
            if v is not None:
                ss[k] = v
    if isinstance(cfg.get("targets"), list):
        picked = [c for c in cfg["targets"] if c in cols]
        if picked:  # 今のCSVに存在する項目が1つもなければ現在の選択を維持
            ss["targets"] = picked
    if isinstance(cfg.get("csv_covs"), list):
        ss["csv_covs"] = [c for c in cfg["csv_covs"] if c in cols]
    sync_state()


num_cols = sync_state()

with st.sidebar:
    st.header(t("sb_header"))
    st.caption(t("sb_caption"))
    txt = json.dumps({"app": "timesfm-forecast", "version": 1, **{k: ss.get(k) for k in CFG_KEYS}},
                     ensure_ascii=False, indent=2)
    st.download_button(t("dl_settings"), txt.encode("utf-8"), "timesfm_settings.json", "application/json")
    if st.button(t("save_local")):
        try:
            SETTINGS_PATH.write_text(txt, encoding="utf-8")
            st.success(t("saved", p=SETTINGS_PATH))
        except OSError as e:
            st.error(t("save_fail", e=e))
    st.divider()
    cfg_file = st.file_uploader(t("pick_cfg"), type=["json"], key="cfg_file")
    b1, b2 = st.columns(2)
    load_from = None
    if b1.button(t("apply_sel"), disabled=cfg_file is None):
        load_from = cfg_file.getvalue()
    if b2.button(t("load_local"), disabled=not SETTINGS_PATH.exists()):
        load_from = SETTINGS_PATH.read_bytes()
    if load_from is not None:
        try:
            apply_cfg(json.loads(load_from.decode("utf-8-sig")))
            num_cols = sync_state()
            st.success(t("applied"))
        except Exception as e:  # noqa: BLE001
            st.error(t("cfg_fail", e=e))

with st.expander(t("preview"), expanded=False):
    st.dataframe(df.head(50), width="stretch")

c1, c2 = st.columns(2)
tcol = c1.selectbox(t("col_time"), cols, key="tcol")
series_col = c2.selectbox(t("series_col"), [""] + [c for c in cols if c != tcol], key="series_col",
                          format_func=lambda x: t("series_none") if x == "" else x, help=t("series_help"))
if not num_cols:
    st.warning(t("no_num", cols=cols))
c5, c6 = st.columns(2)
targets = c5.multiselect(t("col_targets"), num_cols, key="targets")
if series_col:
    series_sel = c6.multiselect(t("series_sel"), series_values(series_col), key="series_sel")
else:
    series_sel = []
    if df[tcol].duplicated().any():
        c6.warning(t("dup_dates"))

c3, c4 = st.columns(2)
val_pct = c3.slider(t("val_pct"), 0, 50, key="val_pct", help=t("val_pct_help"))
horizon = c4.number_input(t("horizon"), min_value=1, max_value=16000, step=1, key="horizon")
from_val = st.checkbox(t("from_val"), key="from_val", help=t("from_val_help"), disabled=val_pct == 0)

with st.expander(t("params_exp")):
    p1, p2 = st.columns(2)
    max_context = p1.number_input(t("max_context"), 32, 16384, step=32, key="max_context",
                                  help=t("max_context_help"))
    normalize = p1.checkbox(t("normalize"), key="normalize")
    qhead = p1.checkbox(t("qhead"), key="qhead")
    flip = p2.checkbox(t("flip"), key="flip")
    positive = p2.checkbox(t("positive"), key="positive")
    fixq = p2.checkbox(t("fixq"), key="fixq")

with st.expander(t("cov_exp"), expanded=False):
    csv_covs = st.multiselect(
        t("csv_covs"),
        [c for c in cols if c not in (tcol, series_col) and c not in targets],
        key="csv_covs",
        help=t("csv_covs_help"),
    )
    cal_opts = st.multiselect(t("cal_opts"), CAL_ALL, key="cal_opts", format_func=lambda x: t("cal:" + x))
    jp_hol = st.checkbox(t("jp_hol"), key="jp_hol")
    fill_mode = st.radio(
        t("fill_mode"), FILL_MODES, horizontal=True, key="fill_mode", help=t("fill_help"),
        format_func=lambda x: t("fill_" + x),
    )
    engine = st.radio(t("engine"), ENGINES, horizontal=True, key="engine", help=t("engine_help"),
                      format_func=lambda x: t("engine_" + x))
    if not jax_ok:
        st.caption(t("jax_caption"))
    use_xreg = engine == "xreg"
    x1, x2 = st.columns(2)
    xmode = x1.radio(t("xmode"), XMODES, horizontal=True, key="xmode")
    ridge = x2.number_input(t("ridge"), 0.0, 1000.0, step=0.1, key="ridge")

if not targets:
    st.warning(t("pick_target"))
    st.stop()
if series_col and not series_sel:
    st.warning(t("pick_series"))
    st.stop()

# ---------------------------------------------------------------- data prep
value_cols = list(targets) + list(csv_covs)
if series_col:
    # ロング形式: (日時, 系列) ごとに1行 → 系列ごと・項目ごとの列（ワイド形式）へ変換
    L = df[[tcol, series_col] + value_cols].copy()
    L[tcol] = pd.to_datetime(L[tcol], errors="coerce")
    L[series_col] = L[series_col].astype(str)
    L = L.dropna(subset=[tcol])
    L = L[L[series_col].isin(series_sel)]
    for c in value_cols:
        if c in targets or to_num(L[c]).notna().mean() > 0.9:
            L[c] = to_num(L[c])
    dates = pd.DatetimeIndex(sorted(L[tcol].unique()))
    d_all = pd.DataFrame(index=dates)
    names, cov_map = [], []
    for c in targets:
        wide = L.groupby([tcol, series_col])[c].last().unstack(series_col)
        for sy in series_sel:
            nm = sy if len(targets) == 1 else f"{sy} | {c}"
            d_all[nm] = wide[sy] if sy in wide.columns else np.nan
            names.append(nm)
            cov_map.append({cv: f"{cv} <{sy}>" for cv in csv_covs})
    for cv in csv_covs:
        wide = L.groupby([tcol, series_col])[cv].last().unstack(series_col)
        for sy in series_sel:
            d_all[f"{cv} <{sy}>"] = wide[sy] if sy in wide.columns else np.nan
    d_all = d_all.rename_axis(tcol).reset_index()
else:
    d_all = df[[tcol] + value_cols].copy()
    d_all[tcol] = pd.to_datetime(d_all[tcol], errors="coerce")
    d_all = d_all.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    for c in targets:
        d_all[c] = to_num(d_all[c])
    names = list(targets)
    cov_map = [{cv: cv for cv in csv_covs} for _ in names]

empty_tail = d_all[names].isna().all(axis=1).to_numpy()
nf = 0  # 末尾の「予測項目が空欄の行」数（＝未来の説明変数が入っている行）
while nf < len(d_all) and empty_tail[len(d_all) - 1 - nf]:
    nf += 1
fut_rows = d_all.iloc[len(d_all) - nf :]
d = d_all.iloc[: len(d_all) - nf].copy().reset_index(drop=True)
los, keep = [], []
for i, c in enumerate(names):
    first = d[c].first_valid_index()
    if first is None:
        continue
    keep.append(i)
    if series_col:  # 系列ごとに開始時期が違うため、先頭の欠損は補完せず開始位置を記録
        d[c] = d[c].interpolate(limit_area="inside").ffill()
        los.append(int(first))
    else:
        d[c] = d[c].interpolate(limit_direction="both")
        los.append(0)
if len(keep) < len(names):
    st.info(t("no_valid", names=", ".join(names[i] for i in range(len(names)) if i not in keep)))
names = [names[i] for i in keep]
cov_map = [cov_map[i] for i in keep]
n = len(d)
if not names or n < 40:
    st.error(t("few_rows"))
    st.stop()
if series_col:
    st.caption(t("long_info", k=len(series_sel), m=len(targets), s=len(names)))
targets = names  # 以降は「系列」の名前リスト（ロング形式なら 系列 x 項目）

# ---------------------------------------------------------------- forecast
if st.button(t("start"), type="primary"):
    v = int(round(n * val_pct / 100))
    if v > 0 and n - v < 32:
        st.error(t("val_too_long"))
        st.stop()
    h = int(horizon)
    cont = bool(from_val and v > 0)  # 検証区間の開始点から (検証+予測) を1回で予測
    mh = math.ceil((v + h if cont else max(v, h)) / 128) * 128
    use_cov = bool(csv_covs or cal_opts)
    eff_xreg = use_xreg and jax_ok
    if use_cov and use_xreg and not jax_ok:
        st.warning(t("xreg_fallback"))
    try:
        fut_idx = future_index(d[tcol], h)
        with st.spinner(t("model_loading")):
            model = get_model(int(max_context), mh, normalize, qhead, flip, positive, fixq, use_cov and eff_xreg)
        timeline = pd.DatetimeIndex(list(d[tcol]) + list(fut_idx))
        S = len(targets)
        cov_num, cov_cat, cache = {}, {}, {}

        def prep_cov(col):
            full = pd.concat([d[col], fut_rows[col]], ignore_index=True)
            if to_num(full).notna().mean() > 0.9:
                x = to_num(full).interpolate(limit_direction="both").reset_index(drop=True)
                if fill_mode == "forecast" and len(x) < n + h:
                    ctx = x.to_numpy(dtype=float)[-int(max_context) :]
                    miss = n + h - len(x)
                    pf = np.asarray(model.forecast(horizon=miss, inputs=[ctx])[0][0])[-miss:]  # backcast付きの場合に備え末尾のみ
                    x = pd.concat([x, pd.Series(pf)], ignore_index=True)
                return "num", x.reindex(range(n + h)).ffill().to_numpy(dtype=float)
            return "cat", full.astype(str).reindex(range(n + h)).ffill().to_numpy()

        for cv in csv_covs:
            per, kind = [], "num"
            for i in range(S):
                col = cov_map[i][cv]
                if col not in cache:
                    cache[col] = prep_cov(col)
                kind, arr = cache[col]
                per.append(arr)
            (cov_num if kind == "num" else cov_cat)[cv] = per
        if csv_covs and nf < h:
            st.warning(t("warn_fill_forecast" if fill_mode == "forecast" else "warn_fill_last", nf=nf))
        cal = calendar_feats(timeline, cal_opts, jp_hol)
        dropped = [k for k, a_ in cal.items() if len(np.unique(a_)) <= 1]
        for k, a_ in cal.items():
            if k not in dropped:
                cov_cat[k] = [a_] * S
        if dropped:
            st.info(t("dropped", names=", ".join(t(k) for k in dropped)))

        def pick(cov, idx):
            return {k: [vv[i] for i in idx] for k, vv in cov.items()}

        with st.spinner(t("predicting")):
            arrs = [d[c].to_numpy(dtype=float) for c in targets]

            def call(idx, end, H):
                return run_forecast(model, [arrs[i] for i in idx], end, H, pick(cov_num, idx), pick(cov_cat, idx),
                                    max_context, xmode, ridge, eff_xreg, [los[i] for i in idx])

            # 検証開始時点で学習データが32行以上ある系列だけ検証する（系列ごとに開始時期が違う場合に備える）
            idx_ok = [i for i in range(S) if v > 0 and n - v - los[i] >= 32]
            if v > 0 and len(idx_ok) < S:
                st.warning(t("val_skipped", names=", ".join(targets[i] for i in range(S) if i not in idx_ok)))
            val_pred = np.full((S, v), np.nan)
            fut_pred = np.zeros((S, h))
            fut_q = np.full((S, h, 10), np.nan)
            val_q = np.full((S, v, 10), np.nan)
            if idx_ok:
                if cont:
                    p, q = call(idx_ok, n - v, v + h)
                    val_pred[idx_ok], fut_pred[idx_ok] = p[:, :v], p[:, v:]
                    if q is not None:
                        fut_q[idx_ok] = q[:, v:, :]
                        val_q[idx_ok] = q[:, :v, :]
                else:
                    p, q = call(idx_ok, n - v, v)
                    val_pred[idx_ok] = p
                    if q is not None:
                        val_q[idx_ok] = q
            idx_fut = [i for i in range(S) if i not in idx_ok] if cont else list(range(S))
            if idx_fut:
                p, q = call(idx_fut, n, h)
                fut_pred[idx_fut] = p
                if q is not None:
                    fut_q[idx_fut] = q
            if np.isnan(fut_q).any():
                fut_q = None
            if v == 0 or not idx_ok or not np.isfinite(val_q[idx_ok]).all():
                val_q = None
    except Exception as e:  # noqa: BLE001
        st.error(t("fail", e=e))
        st.stop()

    metrics = []
    if v > 0:
        for k, c in enumerate(targets):
            if np.isnan(val_pred[k]).all():
                continue
            y, p = arrs[k][n - v :], val_pred[k]
            mape = float(np.mean(np.abs((y - p)[y != 0] / y[y != 0])) * 100) if np.any(y != 0) else np.nan
            metrics.append({"item": c, "MAE": np.mean(np.abs(y - p)), "RMSE": np.sqrt(np.mean((y - p) ** 2)), "MAPE(%)": mape})
    st.session_state["res"] = dict(
        targets=list(targets), v=v, val_pred=val_pred, fut_pred=fut_pred, fut_q=fut_q, val_q=val_q,
        fut_idx=fut_idx, metrics=metrics, tcol=tcol, cont=cont, cov_used=list(cov_num) + list(cov_cat),
    )

res = st.session_state.get("res")
if res and (res["targets"] != list(targets) or res["tcol"] != tcol):
    res = None  # 項目や日時列を変えた場合は古い結果を表示しない

# ---------------------------------------------------------------- graph
st.subheader(t("graph"))
g1, g2, g3 = st.columns([2, 1, 1])
view = g1.radio(t("view"), ["full", "only"], horizontal=True, disabled=res is None, format_func=lambda x: t("view_" + x))
split = g2.checkbox(t("split"), False)
band = g3.checkbox(t("band"), False, disabled=res is None or (res["fut_q"] is None and res.get("val_q") is None))
only = view == "only"

groups = [[c] for c in targets] if split else [targets]
for grp in groups:
    st.plotly_chart(make_fig(grp, targets, d, tcol, res, only, band), width="stretch")

if res:
    if res.get("cont"):
        st.caption(t("cont_note"))
    if res.get("cov_used"):
        st.caption(t("cov_used", names=", ".join(t(n) if n.startswith("cal:") else n for n in res["cov_used"])))
    if res["metrics"]:
        st.subheader(t("val_result"))
        st.dataframe(pd.DataFrame(res["metrics"]).rename(columns={"item": t("col_item")}), width="stretch")
    out = pd.DataFrame({tcol: res["fut_idx"]})
    for k, c in enumerate(res["targets"]):
        out[f"{c}_forecast"] = res["fut_pred"][k]
    st.download_button(t("dl_forecast"), out.to_csv(index=False).encode("utf-8-sig"),
                       "forecast.csv", "text/csv")
