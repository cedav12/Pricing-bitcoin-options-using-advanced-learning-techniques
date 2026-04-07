import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def compute_time_value(df: pd.DataFrame) -> pd.Series:
    if not all(c in df.columns for c in ['option_type', 'underlying_price', 'strike', 'option_price']):
        return pd.Series(np.nan, index=df.index)

    otype = df['option_type'].values
    und = df['underlying_price'].values
    strike = df['strike'].values
    price = df['option_price'].values

    valid_und = np.maximum(und, 1e-8)

    is_call = (otype == 'call') | (otype == 'C')
    is_put = (otype == 'put') | (otype == 'P')

    intrinsic_usd = np.zeros_like(und)
    intrinsic_usd[is_call] = np.maximum(und[is_call] - strike[is_call], 0)
    intrinsic_usd[is_put] = np.maximum(strike[is_put] - und[is_put], 0)

    intrinsic_btc = intrinsic_usd / valid_und
    time_value = price - intrinsic_btc
    return pd.Series(time_value, index=df.index)

def create_buckets(series: pd.Series, method: str, num_bins: int = None, edges=None, format_labels=False):
    s = series.dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype='category'), []
        
    actual_edges = []
    
    if method == "explicit" and edges is not None:
        try:
            edges = [float(e) for e in edges]
        except (ValueError, TypeError):
            pass
            
        if format_labels:
            # Generate readable labels like '1', '2', '3-5'
            labels = []
            for i in range(len(edges)-1):
                raw_low = edges[i]
                raw_high = edges[i+1]
                
                low = int(raw_low) if raw_low.is_integer() else raw_low
                high = int(raw_high) if raw_high.is_integer() else raw_high
                
                if low == 0 and high == 1:
                    labels.append("1")
                elif low == 0:
                    labels.append(f"<={high}")
                elif high == float('inf'):
                    labels.append(f">{low}")
                elif high - low <= 1:
                    labels.append(f"{high}")
                else:
                    labels.append(f"{low+1}-{high}")
                    
            out = pd.cut(series, bins=edges, labels=labels)
        else:
            out = pd.cut(series, bins=edges, labels=False)
        actual_edges = edges
    elif method == "quantile":
        out, bins = pd.qcut(series, q=num_bins, labels=False, duplicates='drop', retbins=True)
        actual_edges = bins.tolist()
    else: # equal_width
        out, bins = pd.cut(series, bins=num_bins, labels=False, retbins=True)
        actual_edges = bins.tolist()
        
    return out, actual_edges

def extract_edges_arg(arg_val):
    if not arg_val:
        return None
    if isinstance(arg_val, list):
        return arg_val
    if isinstance(arg_val, str):
        return [float(x.strip()) for x in arg_val.split(',')]
    return None

# =================================================================================
# LAYER 1: INTEGRITY CHECKS
# =================================================================================

def get_basic_overview(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    unique_ts = df['timestamp'].nunique() if 'timestamp' in df.columns else np.nan
    
    unique_instruments = np.nan
    if all(c in df.columns for c in ['strike', 'expiry', 'option_type']):
        unique_instruments = df[['strike', 'expiry', 'option_type']].drop_duplicates().shape[0]

    ts_min = df['timestamp'].min() if 'timestamp' in df.columns else np.nan
    ts_max = df['timestamp'].max() if 'timestamp' in df.columns else np.nan

    calls = (df['option_type'].isin(['call', 'C'])).sum() if 'option_type' in df.columns else np.nan
    puts = (df['option_type'].isin(['put', 'P'])).sum() if 'option_type' in df.columns else np.nan

    scalars_dict = {
        'total_rows': total_rows,
        'unique_timestamps': unique_ts,
        'unique_instruments': unique_instruments,
        'timestamp_min': ts_min,
        'timestamp_max': ts_max,
        'calls_count': calls,
        'puts_count': puts,
    }
    
    numeric_cols = ['option_price', 'underlying_price', 'strike', 'time_to_maturity', 
                    'volume', 'trade_count', 'realized_variance', 'rolling_std_24h']
    available_cols = [c for c in numeric_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    desc_df = pd.DataFrame()
    if available_cols:
        desc_df = df[available_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        desc_df.reset_index(inplace=True)
        desc_df.rename(columns={'index': 'metric'}, inplace=True)
        
    return {
        "scalars": pd.DataFrame([scalars_dict]),
        "quantiles": desc_df
    }

def get_integrity_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    total_rows = len(df)
    
    for c in df.columns:
        nans = df[c].isna().sum()
        if nans > 0:
            records.append({'check_type': 'missing_values', 'column': c, 'count': nans, 'percentage': nans / total_rows})

    dup_row = df.duplicated().sum()
    records.append({'check_type': 'duplicates', 'column': 'ALL_ROWS', 'count': dup_row, 'percentage': dup_row / total_rows})
    
    key_cols = ['timestamp', 'strike', 'expiry', 'option_type']
    if all(c in df.columns for c in key_cols):
        dup_keys = df.duplicated(subset=key_cols).sum()
        records.append({'check_type': 'duplicate_keys', 'column': '-'.join(key_cols), 'count': dup_keys, 'percentage': dup_keys / total_rows})

    checks = {
        'option_price': ('invalid_nonpositive_option_price', lambda s: s <= 0),
        'underlying_price': ('invalid_nonpositive_underlying_price', lambda s: s <= 0),
        'strike': ('invalid_nonpositive_strike', lambda s: s <= 0),
        'time_to_maturity': ('invalid_nonpositive_time_to_maturity', lambda s: s <= 0),
        'volume': ('invalid_negative_volume', lambda s: s < 0),
        'trade_count': ('invalid_negative_trade_count', lambda s: s < 0),
    }

    for col, (label, cond) in checks.items():
        if col in df.columns:
            invalid_sum = cond(df[col]).sum()
            records.append({'check_type': label, 'column': col, 'count': invalid_sum, 'percentage': invalid_sum / total_rows})

    out = pd.DataFrame(records)
    if out.empty:
        out = pd.DataFrame(columns=['check_type', 'column', 'count', 'percentage'])
    return out

# =================================================================================
# LAYER 2: TRADE-BASED QUALITY DIAGNOSTICS
# =================================================================================

def evaluate_stale_prices(df: pd.DataFrame) -> pd.Series:
    required = ['strike', 'expiry', 'option_type', 'timestamp', 'option_price']
    if not all(c in df.columns for c in required):
        return pd.Series(False, index=df.index)
        
    df_sorted = df[required].sort_values(['strike', 'expiry', 'option_type', 'timestamp'])
    
    same_instr = (
        (df_sorted['strike'] == df_sorted['strike'].shift(1)) &
        (df_sorted['expiry'] == df_sorted['expiry'].shift(1)) &
        (df_sorted['option_type'] == df_sorted['option_type'].shift(1))
    )
    
    is_stale = same_instr & (df_sorted['option_price'] == df_sorted['option_price'].shift(1))
    return is_stale.reindex(df.index, fill_value=False)

def get_stale_price_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = ['strike', 'expiry', 'option_type', 'timestamp', 'option_price']
    if not all(c in df.columns for c in required):
        return pd.DataFrame({'status': ['Missing columns for stale check']})
        
    df_sorted = df[required].sort_values(['strike', 'expiry', 'option_type', 'timestamp'])
    same_instr = (
        (df_sorted['strike'] == df_sorted['strike'].shift(1)) &
        (df_sorted['expiry'] == df_sorted['expiry'].shift(1)) &
        (df_sorted['option_type'] == df_sorted['option_type'].shift(1))
    )
    
    is_stale = same_instr & (df_sorted['option_price'] == df_sorted['option_price'].shift(1))
    total_stale = is_stale.sum()
    total_eval = same_instr.sum()
    
    if total_eval == 0:
        return pd.DataFrame({'stale_rate': [np.nan]})
        
    breaks = (~is_stale).cumsum()
    streaks = is_stale.groupby(breaks).sum()
    valid_streaks = streaks[streaks > 0]
    
    mean_streak = valid_streaks.mean() if not valid_streaks.empty else 0
    max_streak = valid_streaks.max() if not valid_streaks.empty else 0
    
    return pd.DataFrame([{
        'total_stale_observations': total_stale,
        'stale_rate_overall': total_stale / len(df),
        'stale_rate_within_instrument': total_stale / total_eval,
        'mean_stale_streak': mean_streak,
        'max_stale_streak': max_streak
    }])


def get_trade_activity_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    total = len(df)
    
    if 'trade_count' in df.columns:
        tc = df['trade_count'].fillna(-1)
        z_trades = (tc == 0).sum()
        one_trades = (tc == 1).sum()
        two_or_less = (tc <= 2).sum()
        
        records.append({'metric': 'trade_count == 0', 'count': z_trades, 'share': z_trades / total})
        records.append({'metric': 'trade_count == 1', 'count': one_trades, 'share': one_trades / total})
        records.append({'metric': 'trade_count <= 2', 'count': two_or_less, 'share': two_or_less / total})
        
        records.append({'metric': 'trade_count_mean', 'count': np.nan, 'share': df['trade_count'].mean()})
        records.append({'metric': 'trade_count_median', 'count': np.nan, 'share': df['trade_count'].median()})
        
    if 'volume' in df.columns:
        records.append({'metric': 'volume_mean', 'count': np.nan, 'share': df['volume'].mean()})
        records.append({'metric': 'volume_median', 'count': np.nan, 'share': df['volume'].median()})

    return pd.DataFrame(records)


def get_time_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    if 'time_value' not in df.columns:
        tv = compute_time_value(df)
    else:
        tv = df['time_value']
    
    valid_tv = tv.dropna()
    total = len(valid_tv)
    if total == 0:
        return pd.DataFrame()
        
    lt_zero = (valid_tv < 0).sum()
    lt_1e4 = (valid_tv < -1e-4).sum()
    lt_1e3 = (valid_tv < -1e-3).sum()

    records = [
        {'threshold': '< 0', 'count': lt_zero, 'share': lt_zero / total},
        {'threshold': '< -1e-4', 'count': lt_1e4, 'share': lt_1e4 / total},
        {'threshold': '< -1e-3', 'count': lt_1e3, 'share': lt_1e3 / total},
    ]
    return pd.DataFrame(records)

# =================================================================================
# LAYER 3: MODEL-READINESS DIAGNOSTICS & BUCKETING
# =================================================================================

def get_grouped_report(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    if df.empty or not group_cols:
        return pd.DataFrame()
        
    has_tc = 'trade_count' in df.columns
    has_vol = 'volume' in df.columns
    has_price = 'option_price' in df.columns
    has_ts = 'timestamp' in df.columns
    has_k = 'strike' in df.columns
    has_exp = 'expiry' in df.columns
    has_stale = 'is_stale' in df.columns

    if has_tc: 
        is_zero_trade = (df['trade_count'] == 0).astype(int)
        is_single_trade = (df['trade_count'] == 1).astype(int)
        is_le_2_trade = (df['trade_count'] <= 2).astype(int)
    else:
        is_zero_trade, is_single_trade, is_le_2_trade = None, None, None

    is_neg_tv = (df['time_value'] < 0).astype(int)

    agg_funcs = {
        'timestamp': ['count', 'nunique'] if has_ts else ['count'],
    }
    if has_k: agg_funcs['strike'] = 'nunique'
    if has_exp: agg_funcs['expiry'] = 'nunique'
    if has_tc: 
        agg_funcs['trade_count'] = ['median', 'mean']
        df['_is_zero_trade'] = is_zero_trade
        df['_is_single_trade'] = is_single_trade
        df['_is_le_2_trade'] = is_le_2_trade
        agg_funcs['_is_zero_trade'] = 'mean'
        agg_funcs['_is_single_trade'] = 'mean'
        agg_funcs['_is_le_2_trade'] = 'mean'
    if has_vol: agg_funcs['volume'] = ['median', 'mean']
    if has_price: agg_funcs['option_price'] = ['median', 'mean']
    
    df['_is_neg_tv'] = is_neg_tv
    agg_funcs['_is_neg_tv'] = 'mean'
    
    if has_stale:
        agg_funcs['is_stale'] = 'mean'
        
    grouped = df.groupby(group_cols).agg(agg_funcs)
    
    grouped.columns = ['_'.join(c).strip('_') for c in grouped.columns.values]
    grouped.reset_index(inplace=True)
    
    rename_mapping = {
        'timestamp_count': 'row_count',
        'timestamp_nunique': 'unique_timestamps',
        'strike_nunique': 'unique_strikes',
        'expiry_nunique': 'unique_expiries',
        'trade_count_median': 'median_trade_count',
        'trade_count_mean': 'mean_trade_count',
        'volume_median': 'median_volume',
        'volume_mean': 'mean_volume',
        'option_price_median': 'median_option_price',
        'option_price_mean': 'mean_option_price',
        '_is_zero_trade_mean': 'zero_trade_rate',
        '_is_single_trade_mean': 'single_trade_rate',
        '_is_le_2_trade_mean': 'trade_count_le_2_rate',
        '_is_neg_tv_mean': 'negative_time_value_rate',
        'is_stale_mean': 'stale_rate'
    }
    grouped.rename(columns=rename_mapping, inplace=True)
    
    total_rows = len(df)
    grouped['share_total'] = grouped['row_count'] / total_rows
    
    cols_to_drop = ['_is_zero_trade', '_is_single_trade', '_is_le_2_trade', '_is_neg_tv']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return grouped


def get_call_clustering_readiness(
    df: pd.DataFrame, 
    mon_col_name: str,
    ttm_col_name: str
) -> pd.DataFrame:
    if df.empty or mon_col_name not in df.columns or ttm_col_name not in df.columns:
        return pd.DataFrame()

    cluster_report = get_grouped_report(df, [mon_col_name, ttm_col_name])
    if cluster_report.empty:
        return cluster_report
    
    def heur(row):
        rc = row.get('row_count', 0)
        ux = row.get('unique_expiries', 0)
        uk = row.get('unique_strikes', 0)
        uts = row.get('unique_timestamps', 0)
        
        if rc < 500 or ux < 2 or uk < 3 or uts < 100:
            return 'weak'
        elif rc > 5000 and ux >= 5 and uk >= 10 and uts >= 1000:
            return 'strong'
        else:
            return 'usable'

    cluster_report['cluster_status'] = cluster_report.apply(heur, axis=1)
    
    if 'underlying_price' in df.columns:
        extra_agg = df.groupby([mon_col_name, ttm_col_name]).agg(
            median_underlying_price=('underlying_price', 'median'),
            mean_underlying_price=('underlying_price', 'mean')
        ).reset_index()
        cluster_report = pd.merge(cluster_report, extra_agg, on=[mon_col_name, ttm_col_name])

    return cluster_report


def get_recommended_filters_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    total = len(df)
    
    def _add(metric, count, action):
        records.append({'metric': metric, 'count': count, 'share': count/total if total > 0 else 0, 'suggested_action': action})
        
    if 'time_to_maturity' in df.columns:
        _add('time_to_maturity <= 0', (df['time_to_maturity'] <= 0).sum(), 'hard_remove')
        
    if 'time_value' in df.columns:
        _add('time_value < -1e-3', (df['time_value'] < -1e-3).sum(), 'hard_remove')
        _add('negative_time_value', (df['time_value'] < 0).sum(), 'soft_flag')
        
    if 'trade_count' in df.columns:
        _add('trade_count == 0', (df['trade_count'] == 0).sum(), 'hard_remove')
        _add('trade_count == 1', (df['trade_count'] == 1).sum(), 'soft_flag')
        _add('trade_count <= 2', (df['trade_count'] <= 2).sum(), 'keep_but_monitor')
        
    if 'option_price' in df.columns:
        _add('very_low_option_price (<1e-4)', (df['option_price'] < 1e-4).sum(), 'soft_flag')
        
    if 'is_stale' in df.columns:
        _add('stale_price', df['is_stale'].sum(), 'soft_flag')
        
    return pd.DataFrame(records)


# =================================================================================
# MAIN PIPELINE
# =================================================================================

def run_descriptives_pipeline(
    input_path: str,
    output_dir: str,
    filter_trade_count_positive: bool = False,
    call_only: bool = False,
    bucket_method: str = "quantile",
    moneyness_bins: int = 3,
    moneyness_edges=None,
    ttm_bins: int = 3,
    ttm_edges=None,
    price_edges=None,
    trade_count_edges=None,
    volume_edges=None,
    run_stale_check: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    num_loaded = len(df)
    
    mon_edges_arg = extract_edges_arg(moneyness_edges)
    ttm_edges_arg = extract_edges_arg(ttm_edges)
    prc_edges_arg = extract_edges_arg(price_edges)
    tc_edges_arg = extract_edges_arg(trade_count_edges)
    vol_edges_arg = extract_edges_arg(volume_edges)
    
    # Fallbacks for liquidity metrics if "explicit" is used without bounding
    if bucket_method == "explicit":
        if tc_edges_arg is None:
            tc_edges_arg = [0, 1, 2, 5, 20, np.inf]
        if vol_edges_arg is None:
            vol_edges_arg = [0, 10, 50, 100, 500, np.inf]
    
    if filter_trade_count_positive and 'trade_count' in df.columns:
        df = df[df['trade_count'] > 0]
        
    if call_only and 'option_type' in df.columns:
        df = df[df['option_type'].isin(['call', 'C'])]

    num_after_filters = len(df)
    print(f"Loaded {num_loaded} rows -> {num_after_filters} after filters.")

    # Always compute time_value for downstream usage
    df['time_value'] = compute_time_value(df)

    if run_stale_check:
        df['is_stale'] = evaluate_stale_prices(df)

    # 1. Integrity and Basics
    print("Running Layer 1: Integrity Checks...")
    overview_dict = get_basic_overview(df)
    overview_dict['scalars'].to_csv(os.path.join(output_dir, "overview_scalars.csv"), index=False)
    if not overview_dict['quantiles'].empty:
        overview_dict['quantiles'].to_csv(os.path.join(output_dir, "overview_quantiles.csv"), index=False)
        
    integ = get_integrity_summary(df)
    integ.to_csv(os.path.join(output_dir, "missingness.csv"), index=False)
    
    # 2. Trade Quality & Missing Reports
    print("Running Layer 2: Quality Diagnostics...")
    trade_act = get_trade_activity_summary(df)
    trade_act.to_csv(os.path.join(output_dir, "trade_activity_summary.csv"), index=False)
    
    tv_sum = get_time_value_summary(df)
    tv_sum.to_csv(os.path.join(output_dir, "time_value_summary.csv"), index=False)
    
    if run_stale_check:
        stales = get_stale_price_summary(df)
        stales.to_csv(os.path.join(output_dir, "stale_price_summary.csv"), index=False)
        
    if 'option_type' in df.columns:
        get_grouped_report(df, ['option_type']).to_csv(os.path.join(output_dir, "by_option_type.csv"), index=False)
        
    # Tracker for explicit outside-bounds metrics
    outside_bounds = []

    # Grouped volumes / trade counts using bucketing
    actual_tc_edges, actual_vol_edges = [], []
    if 'trade_count' in df.columns:
        df['tc_bucket'], actual_tc_edges = create_buckets(df['trade_count'], bucket_method, num_bins=3, edges=tc_edges_arg, format_labels=True)
        if bucket_method == "explicit":
            obs = df['trade_count'].notna().sum()
            binned = df['tc_bucket'].notna().sum()
            outside_bounds.append({'metric': 'trade_count', 'total_obs': obs, 'binned_obs': binned, 'outside_range': obs - binned, 'share_outside': (obs - binned)/obs if obs else 0})
        get_grouped_report(df, ['tc_bucket']).to_csv(os.path.join(output_dir, "by_trade_count_bucket.csv"), index=False)
        
    if 'volume' in df.columns:
        df['vol_bucket'], actual_vol_edges = create_buckets(df['volume'], bucket_method, num_bins=3, edges=vol_edges_arg, format_labels=True)
        if bucket_method == "explicit":
            obs = df['volume'].notna().sum()
            binned = df['vol_bucket'].notna().sum()
            outside_bounds.append({'metric': 'volume', 'total_obs': obs, 'binned_obs': binned, 'outside_range': obs - binned, 'share_outside': (obs - binned)/obs if obs else 0})
        get_grouped_report(df, ['vol_bucket']).to_csv(os.path.join(output_dir, "by_volume_bucket.csv"), index=False)
        
    # 3. Model Readiness / Bucketing
    print("Running Layer 3: Model Readiness Diagnostics...")
    actual_mon_edges, actual_ttm_edges = [], []
    
    if 'log_moneyness' in df.columns:
        bins = 100
        hist_mon = get_histogram_table(df['log_moneyness'], bins=bins)
        hist_mon.to_csv(os.path.join(output_dir, f"hist_moneyness_{bins}bins.csv"), index=False)
        save_histogram_plot(
            df['log_moneyness'],
            os.path.join(output_dir, f"hist_moneyness_{bins}bins.png"),
            title="Distribution of Log Moneyness",
            xlabel="Log Moneyness",
            bins=bins
        )

        df['mon_bin'], actual_mon_edges = create_buckets(df['log_moneyness'], method=bucket_method, num_bins=moneyness_bins, edges=mon_edges_arg)
        if bucket_method == "explicit":
            obs = df['log_moneyness'].notna().sum()
            binned = df['mon_bin'].notna().sum()
            outside_bounds.append({'metric': 'log_moneyness', 'total_obs': obs, 'binned_obs': binned, 'outside_range': obs - binned, 'share_outside': (obs - binned)/obs if obs else 0})
        get_grouped_report(df, ['mon_bin']).to_csv(os.path.join(output_dir, "by_moneyness.csv"), index=False)

    if 'time_to_maturity' in df.columns:
        bins = 100
        hist_ttm = get_histogram_table(df['time_to_maturity'], bins=bins)
        hist_ttm.to_csv(os.path.join(output_dir, f"hist_time_to_maturity_{bins}bins.csv"), index=False)
        save_histogram_plot(
            df['time_to_maturity'],
            os.path.join(output_dir, f"hist_time_to_maturity_{bins}bins.png"),
            title="Distribution of Time to Maturity",
            xlabel="Time to Maturity",
            bins=bins
        )

        df['ttm_bin'], actual_ttm_edges = create_buckets(df['time_to_maturity'], method=bucket_method, num_bins=ttm_bins, edges=ttm_edges_arg)
        if bucket_method == "explicit":
            obs = df['time_to_maturity'].notna().sum()
            binned = df['ttm_bin'].notna().sum()
            outside_bounds.append({'metric': 'time_to_maturity', 'total_obs': obs, 'binned_obs': binned, 'outside_range': obs - binned, 'share_outside': (obs - binned)/obs if obs else 0})
        get_grouped_report(df, ['ttm_bin']).to_csv(os.path.join(output_dir, "by_maturity.csv"), index=False)

    if outside_bounds and bucket_method == "explicit":
        pd.DataFrame(outside_bounds).to_csv(os.path.join(output_dir, "outside_explicit_bounds_summary.csv"), index=False)

    if 'mon_bin' in df.columns and 'ttm_bin' in df.columns:
        get_grouped_report(df, ['mon_bin', 'ttm_bin']).to_csv(os.path.join(output_dir, "by_moneyness_x_maturity.csv"), index=False)
        
        if 'tc_bucket' in df.columns:
            get_grouped_report(df, ['tc_bucket', 'mon_bin']).to_csv(os.path.join(output_dir, "by_trade_count_x_moneyness.csv"), index=False)
            get_grouped_report(df, ['tc_bucket', 'ttm_bin']).to_csv(os.path.join(output_dir, "by_trade_count_x_maturity.csv"), index=False)

    # 3x3 heuristic explicitly on calls using the current buckets
    if call_only:
        calls = df
    else:
        calls = df[df['option_type'].isin(['call', 'C'])] if 'option_type' in df.columns else df
        
    if 'mon_bin' in calls.columns and 'ttm_bin' in calls.columns:
        call_3x3 = get_call_clustering_readiness(
            calls, mon_col_name='mon_bin', ttm_col_name='ttm_bin'
        )
        if not call_3x3.empty:
            call_3x3.to_csv(os.path.join(output_dir, "call_3x3_cluster_summary.csv"), index=False)

    # Output Recommended Filters
    print("Generating recommended filters summary...")
    rec_filters = get_recommended_filters_summary(df)
    rec_filters.to_csv(os.path.join(output_dir, "recommended_filters_summary.csv"), index=False)

    # Dump Metadata
    metadata = {
        'input_path': input_path,
        'rows_loaded': num_loaded,
        'rows_after_filters': num_after_filters,
        'call_only_applied': call_only,
        'trade_count_positive_applied': filter_trade_count_positive,
        'stale_check_run': run_stale_check,
        'bucketing_method_used': bucket_method,
        'actual_moneyness_edges': actual_mon_edges,
        'actual_ttm_edges': actual_ttm_edges,
        'actual_trade_count_edges': actual_tc_edges,
        'actual_volume_edges': actual_vol_edges
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as jf:
        json.dump(metadata, jf, indent=4)
        
    print(f"Finished successfully. Outputs saved to -> {output_dir}")


def get_histogram_table(series: pd.Series, bins: int = 10) -> pd.DataFrame:
    s = pd.to_numeric(series, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return pd.DataFrame(columns=['bin_left', 'bin_right', 'count', 'share'])

    counts, edges = np.histogram(s, bins=bins)
    total = counts.sum()

    return pd.DataFrame({
        'bin_left': edges[:-1],
        'bin_right': edges[1:],
        'count': counts,
        'share': counts / total if total > 0 else 0
    })


def save_histogram_plot(series: pd.Series, output_path: str, title: str, xlabel: str, bins: int = 10):
    s = pd.to_numeric(series, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
