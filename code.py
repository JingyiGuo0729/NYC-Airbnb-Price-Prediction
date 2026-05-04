#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import numpy as np
import pandas as pd

DATA_PATH = "D:\\下载\\Airbnb_Open_Data.csv\\Airbnb_Open_Data.csv"

def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def parse_money(x):
    """把 '$1,234' / '1,234' / None 转成 float；转不动返回 NaN"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_bool_str(x):
    """把 'True'/'False'/True/False/1/0 等统一成 0/1"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    s = str(x).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return 1
    if s in {"false", "f", "no", "n", "0"}:
        return 0
    return np.nan

# 1) 读取
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
df = df_raw.copy()

# 2) 统一列名（snake_case）
df.columns = [to_snake(c) for c in df.columns]

# 3) 统一字符串：去首尾空格
obj_cols = df.select_dtypes(include="object").columns
for c in obj_cols:
    df[c] = df[c].astype("string").str.strip()

# 4) 关键字段类型转换
# 日期
if "last_review" in df.columns:
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

# 金额：保留原字段，同时新增数值字段
# （文档中提到要对 money 字段去符号、去逗号再数值化）
if "price" in df.columns:
    df["price_num"] = df["price"].map(parse_money)

if "service_fee" in df.columns:
    df["service_fee_num"] = df["service_fee"].map(parse_money)

# 布尔/二值
if "instant_bookable" in df.columns:
    df["instant_bookable_01"] = df["instant_bookable"].map(parse_bool_str)

# host_identity_verified 常见取值：'verified'/'unconfirmed'/缺失
if "host_identity_verified" in df.columns:
    df["host_identity_verified_01"] = (
        df["host_identity_verified"]
        .astype("string").str.lower()
        .map({"verified": 1, "unconfirmed": 0})
    )

# 5) 去重（行完全重复）
df = df.drop_duplicates().reset_index(drop=True)

# 6) 明显不合理值（保守处理：转 NaN，后续第2章统一处理缺失）
for col, lo, hi in [
    ("lat", -90, 90),
    ("long", -180, 180),
    ("construction_year", 1700, 2025),
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

print("第1章完成：df.shape =", df.shape)


# In[3]:


# =========================
# 第2章：缺失处理（指示变量 + 业务一致填补 + 派生特征）
# =========================
import numpy as np
import pandas as pd

def missing_report(data: pd.DataFrame) -> pd.DataFrame:
    mr = data.isna().mean().sort_values(ascending=False)
    out = pd.DataFrame({"missing_rate": mr, "missing_cnt": data.isna().sum()})
    return out

# 1) 缺失概览（可打印/保存）
mr = missing_report(df)
print(mr.head(20))

# 2) 近乎全缺失字段：license（文档中提到缺失率极高可直接剔除）
if "license" in df.columns:
    if df["license"].isna().mean() > 0.99:
        df = df.drop(columns=["license"])

# 3) 缺失指示变量（按“缺失即信息”的思路）
indicator_cols = []
for c in ["host_response_time", "house_rules", "last_review", "reviews_per_month",
          "construction_year", "host_identity_verified"]:
    if c in df.columns:
        newc = f"{c}__isna"
        df[newc] = df[c].isna().astype(int)
        indicator_cols.append(newc)

if "number_of_reviews" in df.columns:
    df["number_of_reviews"] = pd.to_numeric(df["number_of_reviews"], errors="coerce")

df["has_reviews"] = np.where(df.get("number_of_reviews", 0).fillna(0) > 0, 1, 0)

if "reviews_per_month" in df.columns:
    df["reviews_per_month"] = pd.to_numeric(df["reviews_per_month"], errors="coerce")
    df["reviews_per_month_filled"] = df["reviews_per_month"].where(df["has_reviews"] == 1, 0.0)
    # 对于 has_reviews==1 但 rpm 仍缺失：用同列中位数兜底
    med_rpm = df.loc[df["has_reviews"] == 1, "reviews_per_month"].median()
    df.loc[(df["has_reviews"] == 1) & (df["reviews_per_month_filled"].isna()), "reviews_per_month_filled"] = med_rpm

if "last_review" in df.columns:
    # 参考日期：用数据中最大 last_review（更稳健）
    ref_date = df["last_review"].max()
    df["days_since_last_review"] = (ref_date - df["last_review"]).dt.days
    df.loc[df["has_reviews"] == 0, "days_since_last_review"] = -1
    df.loc[df["days_since_last_review"].isna(), "days_since_last_review"] = -1

# 5) host_response_time：缺失单独作为一类（Unknown）
if "host_response_time" in df.columns:
    df["host_response_time"] = df["host_response_time"].fillna("Unknown")

# 6) 金额字段：缺失量很小但建议用“分组中位数”填补（neighbourhood_group + room_type）
group_keys = [c for c in ["neighbourhood_group", "room_type"] if c in df.columns]

def group_median_impute(data: pd.DataFrame, col: str, keys):
    if col not in data.columns:
        return
    data[col] = pd.to_numeric(data[col], errors="coerce")
    if not keys:
        data[col] = data[col].fillna(data[col].median())
        return
    gmed = data.groupby(keys)[col].transform("median")
    data[col] = data[col].fillna(gmed)
    data[col] = data[col].fillna(data[col].median())

group_median_impute(df, "service_fee_num", group_keys)

# 7) house_rules：缺失多 => 先填空串，再做简单文本派生特征（下一章/下一节也可继续扩展）
if "house_rules" in df.columns:
    df["house_rules"] = df["house_rules"].fillna("")
    txt = df["house_rules"].astype("string").fillna("")
    df["house_rules_len"] = txt.str.len()
    df["house_rules_words"] = txt.str.split().map(lambda x: len(x) if isinstance(x, list) else 0)

    low = txt.str.lower()
    df["rule_no_smoking"] = low.str.contains("no smoking|smoking not allowed|禁止吸烟", regex=True).astype(int)
    df["rule_no_pets"]    = low.str.contains("no pets|pets not allowed|禁止携带宠物", regex=True).astype(int)
    df["rule_no_party"]   = low.str.contains("no party|parties not allowed|禁止聚会", regex=True).astype(int)

print("第2章完成：df.shape =", df.shape)


# In[4]:


# =========================
# 第3章：Box-Cox 偏态矫正 + 异常值截断（k=3）
# =========================
import numpy as np
import pandas as pd
from scipy import stats

# ---- 3.1 Box-Cox（按文档给定 λ 与 shift）
# 文档给定的参数（表3-3）：lambda 与 shift（其中 number_of_reviews/availability_365 需要 shift=1）
BOXCOX_PARAMS = {
    "minimum_nights": (0.432177275, 0),
    "calculated_host_listings_count": (-1.087900293, 0),
    "reviews_per_month": (0.134958571, 0),
    "number_of_reviews": (-0.081709301, 1),
    "availability_365": (0.281646573, 1),
}

def boxcox_apply(series: pd.Series, lmbda: float, shift: float, out_name: str):
    x = pd.to_numeric(series, errors="coerce").astype(float)
    # 仅对非缺失做变换；缺失保留，后面用 filled 版本衔接建模
    mask = x.notna()
    x2 = x.copy()

    # 基础 shift
    x2.loc[mask] = x2.loc[mask] + shift

    # 若仍出现 <=0（比如你第2章把 reviews_per_month_filled 设为0），额外加一个极小量兜底
    minv = x2.loc[mask].min()
    if minv <= 0:
        eps = abs(minv) + 1e-6
        x2.loc[mask] = x2.loc[mask] + eps

    y = pd.Series(np.nan, index=x.index, dtype=float)
    y.loc[mask] = stats.boxcox(x2.loc[mask], lmbda=lmbda)
    return y

for col, (lmbda, shift) in BOXCOX_PARAMS.items():
    if col in df.columns:
        df[f"{col}__bc"] = boxcox_apply(df[col], lmbda=lmbda, shift=shift, out_name=f"{col}__bc")

# 对 reviews_per_month 的 Box-Cox：建议建模用一个“填补后的版本”
if "reviews_per_month__bc" in df.columns:
    df["reviews_per_month__bc_filled"] = df["reviews_per_month__bc"].fillna(0.0)

# ---- 3.2 异常值处理：均值±3σ + 截断（clip）
OUTLIER_COLS = [c for c in [
    "minimum_nights",
    "calculated_host_listings_count",
    "reviews_per_month_filled",
    "number_of_reviews",
    "availability_365",
] if c in df.columns]

def clip_by_mean_std(data: pd.DataFrame, col: str, k: float = 3.0):
    x = pd.to_numeric(data[col], errors="coerce").astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    lo, hi = mu - k * sd, mu + k * sd
    # 标注异常（可用于报告/对比）
    data[f"{col}__outlier"] = ((x < lo) | (x > hi)).astype(int)
    # 截断
    data[col] = x.clip(lower=lo, upper=hi)
    return lo, hi

bounds = {}
for c in OUTLIER_COLS:
    lo, hi = clip_by_mean_std(df, c, k=3.0)
    bounds[c] = (lo, hi)

print("第3章完成：截断边界（部分展示）")
for k, v in list(bounds.items())[:5]:
    print(k, "=>", v)


# In[5]:


import os
import numpy as np
import pandas as pd

OUT_DIR = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

BIN_K = 5

# ---------- 1) 等频分箱（修复版：避免 Categorical 赋值冲突）
def qcut_with_missing(s: pd.Series, q=5, missing_label="Missing"):
    """
    返回一个带 Missing 类别的有序 Categorical Series
    修复：不在空 Categorical 上做 loc 赋值，避免 categories 不一致报错
    """
    s_num = pd.to_numeric(s, errors="coerce")
    mask = s_num.notna()

    out = pd.Series(missing_label, index=s.index, dtype="object")

    if mask.sum() > 0:
        bins = pd.qcut(s_num.loc[mask], q=q, duplicates="drop")  # Categorical[Interval]
        out.loc[mask] = bins.astype(object)                      # 保存 Interval 对象
        cats = list(bins.cat.categories) + [missing_label]
        out = pd.Categorical(out, categories=cats, ordered=True)
    else:
        out = pd.Categorical(out, categories=[missing_label], ordered=True)

    return pd.Series(out, index=s.index)

# 要分箱的列（存在就做）
BIN_COLS = [c for c in [
    "number_of_reviews",
    "calculated_host_listings_count",
    "reviews_per_month",
    "minimum_nights"
] if c in df.columns]

# 1.1 生成分箱类别 + 有序标签编码（Missing=-1）
for c in BIN_COLS:
    df[f"{c}__bin_cat"] = qcut_with_missing(df[c], q=BIN_K, missing_label="Missing")

    cat = df[f"{c}__bin_cat"]
    # 非 Missing 的区间类别（按从小到大）
    levels = [x for x in cat.cat.categories if x != "Missing"]
    level2id = {lv: i for i, lv in enumerate(levels)}

    df[f"{c}__bin_id"] = cat.map(lambda x: level2id.get(x, -1)).astype(int)

# 1.2 （可选）One-Hot：用于线性模型 / 神经网络；树模型可直接用 __bin_id
bin_cat_cols = [f"{c}__bin_cat" for c in BIN_COLS]
bin_onehot = pd.get_dummies(df[bin_cat_cols], prefix=[c.replace("__bin_cat", "") for c in bin_cat_cols])

if "review_rate_number" in df.columns and "price_num" in df.columns:
    rr = pd.to_numeric(df["review_rate_number"], errors="coerce")
    df["rating_level"] = rr.round().clip(1, 5).astype("Int64")

    y = pd.to_numeric(df["price_num"], errors="coerce")
    rank = y.rank(method="average", na_option="keep")
    n = rank.notna().sum()

    # 每个 rating_level 的平均秩 / n -> [0,1]
    midrank_map = (
        pd.DataFrame({"rating_level": df["rating_level"], "rank": rank})
        .dropna()
        .groupby("rating_level")["rank"]
        .mean()
        .sort_index()
    )
    midrank_score = (midrank_map / n).to_dict()

    df["review_rate_midrank"] = df["rating_level"].map(midrank_score).astype(float)

# ---------- 3) hybrid 合并：Top20 + 其他 + 缺失
def topn_other(data: pd.DataFrame, col: str, topn=20, other_label="其他", missing_label="缺失"):
    s = data[col].astype("string")
    top = s.value_counts(dropna=True).head(topn).index
    out = s.where(s.isin(top), other_label)
    out = out.fillna(missing_label)
    return out

for col in ["price", "house_rules", "neighbourhood"]:
    if col in df.columns:
        df[f"{col}__top20"] = topn_other(df, col, topn=20, other_label="其他", missing_label="缺失")

# ---------- 4) 生成第4章结束版特征矩阵 X（避免价格泄露：删掉 price / price_num）
DROP_FOR_X = [c for c in ["price", "price_num"] if c in df.columns]
X_base = df.drop(columns=DROP_FOR_X, errors="ignore")

# 拼上分箱 one-hot（如果你不想 one-hot，就把这行换成：X_ch4 = X_base.copy()）
X_ch4 = pd.concat([X_base, bin_onehot], axis=1)

# ---------- 5) 导出（Windows / Mac / Linux 都能跑）
df_path = os.path.join(OUT_DIR, "airbnb_after_ch4.csv")
x_path  = os.path.join(OUT_DIR, "X_after_ch4.csv")

df.to_csv(df_path, index=False, encoding="utf-8-sig")
X_ch4.to_csv(x_path, index=False, encoding="utf-8-sig")

print("第4章完成！")
print("df.shape   =", df.shape, "->", df_path)
print("X_ch4.shape=", X_ch4.shape, "->", x_path)


# In[6]:


import os
import numpy as np
import pandas as pd

OUT_DIR = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# 5.1 低频类别合并（并集规则）
# -------------------------
FREQ_TH = 100
PROP_TH = 0.001

def merge_low_freq(series: pd.Series,
                   freq_th=100,
                   prop_th=0.001,
                   other_label="其他",
                   missing_label="缺失"):
    s = series.astype("string").fillna(missing_label)
    vc = s.value_counts(dropna=False)
    n = vc.sum()
    low_mask = (vc < freq_th) | ((vc / n) < prop_th)
    low_values = set(vc[low_mask].index.tolist())
    return s.where(~s.isin(low_values), other_label)

cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
focus_cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) >= 30]

for c in focus_cat_cols:
    df[f"{c}__lf_merge"] = merge_low_freq(df[c], FREQ_TH, PROP_TH)

print(f"[第5章-5.1] 低频合并列数：{len(focus_cat_cols)}")


# -------------------------
# 5.2 高偏度数值处理（稳健版）
# -------------------------
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

def winsorize_series(x: pd.Series, lower_q=0.001, upper_q=0.999):
    """分位数截断，避免极端值导致溢出/优化失败"""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    if x.notna().sum() == 0:
        return x
    lo = x.quantile(lower_q)
    hi = x.quantile(upper_q)
    return x.clip(lo, hi)

def signed_log1p(x: pd.Series):
    """兼容负数的 log1p：sign(x)*log1p(|x|)"""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return np.sign(x) * np.log1p(np.abs(x))

def safe_skew_transform_column(s: pd.Series, colname: str,
                               winsor_lower=0.001, winsor_upper=0.999,
                               qt_n_quantiles=1000,
                               random_state=42):
    """
    返回：transformed_series, method_name
    逻辑：
    1) winsorize
    2) 尝试 scipy.stats.yeojohnson（自动估λ）
       - 若失败：signed_log1p
       - 再失败：QuantileTransformer(输出正态分布)
    """
    x = pd.to_numeric(s, errors="coerce").astype(float)
    # 常数列/有效值太少：直接返回原值
    valid = x.dropna()
    if valid.size < 50 or valid.nunique() <= 1:
        return x, "skip_constant_or_few"

    # winsorize 先压住极端值
    xw = winsorize_series(x, winsor_lower, winsor_upper)

    # 先试 Yeo-Johnson（scipy）
    try:
        v = xw.dropna().values
        # 直接用 yeojohnson 会内部找最优 λ，比 sklearn 稳一点（我们已经 winsorize 过）
        yt, lmbda = stats.yeojohnson(v)  # 可能仍失败
        out = pd.Series(np.nan, index=xw.index, dtype=float)
        out.loc[xw.notna()] = stats.yeojohnson(xw.dropna().values, lmbda=lmbda)
        return out, f"yeojohnson(lmbda={lmbda:.4f})"
    except Exception:
        pass

    # 再试 signed log1p
    try:
        out = signed_log1p(xw)
        return out, "signed_log1p"
    except Exception:
        pass

    # 最后兜底：QuantileTransformer -> N(0,1)
    try:
        qt = QuantileTransformer(
            n_quantiles=min(qt_n_quantiles, int(valid.size)),
            output_distribution="normal",
            random_state=random_state,
            subsample=int(1e9)  # 尽量不抽样（样本少时也稳）
        )
        arr = xw.values.reshape(-1, 1)
        mask = ~np.isnan(arr[:, 0])
        out = pd.Series(np.nan, index=xw.index, dtype=float)
        out.loc[mask] = qt.fit_transform(arr[mask]).ravel()
        return out, "quantile_to_normal"
    except Exception:
        return xw, "fallback_winsor_only"

# 选数值列（排除 price_num 等可能用于标签的列）
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for dropc in ["price_num"]:
    if dropc in num_cols:
        num_cols.remove(dropc)

# 只挑高偏度列（阈值可调）
SKEW_TH = 1.0
sk = df[num_cols].skew(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
skewed = sk[sk.abs() >= SKEW_TH].index.tolist()

print(f"[第5章-5.2] 高偏度列数量 = {len(skewed)}（阈值 |skew| >= {SKEW_TH}）")

transform_log = []
for c in skewed:
    transformed, method = safe_skew_transform_column(df[c], c,
                                                     winsor_lower=0.001,
                                                     winsor_upper=0.999)
    df[f"{c}__skewfix"] = transformed
    transform_log.append((c, float(sk[c]), method))

transform_log_df = pd.DataFrame(transform_log, columns=["col", "skew", "method"]).sort_values("skew", key=np.abs, ascending=False)
print("[第5章-5.2] 变换方法统计：")
print(transform_log_df["method"].value_counts())
transform_log_df.to_csv(os.path.join(OUT_DIR, "ch5_skew_transform_log.csv"),
                        index=False, encoding="utf-8-sig")


# -------------------------
# 5.3 IQR 异常值：标记 + 截断列
# -------------------------
def iqr_clip(series: pd.Series, k=1.5):
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if x.notna().sum() == 0:
        flag = pd.Series(0, index=x.index, dtype=int)
        return x, flag, np.nan, np.nan

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    flag = ((x < lo) | (x > hi)).astype(int)
    clipped = x.clip(lo, hi)
    return clipped, flag, lo, hi

OUTLIER_CANDIDATES = [c for c in [
    "minimum_nights",
    "number_of_reviews",
    "availability_365",
    "calculated_host_listings_count",
    "reviews_per_month_filled",
] if c in df.columns]

bounds = []
for c in OUTLIER_CANDIDATES:
    clipped, flag, lo, hi = iqr_clip(df[c], k=1.5)
    df[f"{c}__iqr_outlier"] = flag
    df[f"{c}__iqr_clip"] = clipped
    bounds.append((c, lo, hi, int(flag.sum())))

bounds_df = pd.DataFrame(bounds, columns=["col", "iqr_lo", "iqr_hi", "outlier_cnt"])
bounds_df.to_csv(os.path.join(OUT_DIR, "ch5_iqr_bounds.csv"),
                 index=False, encoding="utf-8-sig")

print(f"[第5章-5.3] IQR处理列数：{len(OUTLIER_CANDIDATES)}")


# -------------------------
# 导出第5章结果
# -------------------------
out_path = os.path.join(OUT_DIR, "airbnb_after_ch5.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print("第5章完成：df.shape =", df.shape, "->", out_path)
print("额外导出：ch5_skew_transform_log.csv / ch5_iqr_bounds.csv")


# In[15]:


import os
import pandas as pd

# -------------------------
# 6 数据规约
# -------------------------
INPUT_PATH = r"C:\Users\admin112\大数据预处理\output\airbnb_after_ch5.csv"

OUT_DIR = os.path.join(os.getcwd(), "chapter6_out")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(OUT_DIR, "chapter6_keep21.csv")

keep_cols_21 = [
    "price_num",
    "minimum_nights__skewfix",
    "number_of_reviews__skewfix",
    "reviews_per_month_filled__skewfix",
    "calculated_host_listings_count__skewfix",
    "days_since_last_review__skewfix",
    "house_rules_len__skewfix",
    "long__skewfix",
    "lat",
    "construction_year",
    "review_rate_midrank",
    "instant_bookable_01",
    "host_identity_verified_01",
    "has_reviews",
    "rule_no_smoking",
    "rule_no_pets",
    "rule_no_party",
    "room_type",
    "neighbourhood_group",
    "cancellation_policy",
    "neighbourhood__lf_merge",
]

df = pd.read_csv(INPUT_PATH, low_memory=False)

if "price_num" not in df.columns:
    if "price" in df.columns:
        df["price_num"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
            errors="coerce"
        )
    else:
        print("原数据中没有 price_num，也没有 price，无法生成 price_num。")

existing = [c for c in keep_cols_21 if c in df.columns]
missing = [c for c in keep_cols_21 if c not in df.columns]

df21 = df[existing].copy()
df21.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("已导出 21 列CSV：", OUT_PATH)
print("保留列数：", len(existing), "/", len(keep_cols_21))
if missing:
    print("以下列在数据中不存在（未能导出）：")
    for c in missing:
        print(" -", c)


# In[1]:


import os
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RULE_CSV_PATH = r"C:\Users\admin112\大数据预处理\chapter6_out\chapter6_keep21.csv"
df = pd.read_csv(RULE_CSV_PATH, low_memory=False)

final_cols_21 = [
    "price_num",
    "minimum_nights__skewfix",
    "number_of_reviews__skewfix",
    "reviews_per_month_filled__skewfix",
    "calculated_host_listings_count__skewfix",
    "days_since_last_review__skewfix",
    "house_rules_len__skewfix",
    "long__skewfix",
    "lat",
    "construction_year",
    "review_rate_midrank",
    "instant_bookable_01",
    "host_identity_verified_01",
    "has_reviews",
    "rule_no_smoking",
    "rule_no_pets",
    "rule_no_party",
    "room_type",
    "neighbourhood_group",
    "cancellation_policy",
    "neighbourhood__lf_merge",
]

use_cols = [c for c in final_cols_21 if c in df.columns]
df21 = df[use_cols].copy()

if "price_num" not in df21.columns:
    raise ValueError("缺少 price_num，请检查 6.3 输出。")

df21 = df21.copy()
df21["price_num"] = pd.to_numeric(df21["price_num"], errors="coerce")
before = len(df21)
df21 = df21[df21["price_num"].notna()].copy()
after = len(df21)
y = df21["price_num"]
X = df21.drop(columns=["price_num"])


print("只使用 21 列后的形状：", df21.shape)

def compress_topk_one_col(s: pd.Series, topk=20, other="__OTHER__", missing="__MISSING__"):
    s = s.astype("object")
    s = s.where(s.notna(), missing)
    vc = s.value_counts(dropna=False)
    kept = set(vc.head(topk).index.tolist())  # 保留最常见 topk 个
    return s.where(s.isin(kept), other)

#压缩
TOPK_NEIGH = 20   

X2 = X.copy()
if "neighbourhood__lf_merge" in X2.columns:
    before = X2["neighbourhood__lf_merge"].astype("object").nunique(dropna=True)
    X2["neighbourhood__lf_merge"] = compress_topk_one_col(X2["neighbourhood__lf_merge"], topk=TOPK_NEIGH)
    after = X2["neighbourhood__lf_merge"].astype("object").nunique(dropna=False)
    other_rate = (X2["neighbourhood__lf_merge"] == "__OTHER__").mean()
    print(f"neighbourhood__lf_merge 压缩：unique {before} -> {after} | OTHER占比={other_rate:.2%}")
else:
    print("X里没有 neighbourhood__lf_merge，跳过压缩。")

# one-hot列数
cat_cols = [c for c in ["room_type", "neighbourhood_group", "cancellation_policy", "neighbourhood__lf_merge"] if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

def onehot_dim_count(Xdf, num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    Z = pre.fit_transform(Xdf.sample(n=min(len(Xdf), 20000), random_state=42) if len(Xdf) > 20000 else Xdf)
    return Z.shape[1]

p_before_compress = onehot_dim_count(X,  num_cols, cat_cols)
p_after_compress  = onehot_dim_count(X2, num_cols, cat_cols)

print(f"one-hot 后列数（压缩前）p = {p_before_compress}")
print(f"one-hot 后列数（压缩后）p = {p_after_compress}")

#Lasso（压缩后）
y_lasso = np.log1p(y) 

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))  
        ]), num_cols),

        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ]), cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

lasso = LassoCV(
    alphas=np.logspace(-2, 1, 25),
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    max_iter=3000,
    tol=1e-3,
    selection="random",
    n_jobs=-1,
    random_state=42
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("lasso", lasso)
])

pipe.fit(X2, y_lasso)
print("Lasso 训练完成。最佳 alpha =", pipe.named_steps["lasso"].alpha_)


# In[2]:


# 取出 one-hot + 数值拼接后的特征名
feat_names = pipe.named_steps["preprocess"].get_feature_names_out()
print("最终进入 Lasso 的特征数：", len(feat_names))
print(feat_names)


# In[3]:


import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

KEEP21_PATH = r"C:\Users\admin112\大数据预处理\chapter6_out\chapter6_keep21.csv"

OUT_DIR = os.path.join(os.getcwd(), "chapter6_out")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TABLE = os.path.join(OUT_DIR, "chapter6_ready_table.csv")       # 压缩后
OUT_ONEHOT = os.path.join(OUT_DIR, "chapter6_ready_onehot51.csv")   # one-hot后

df = pd.read_csv(KEEP21_PATH, low_memory=False)
df["price_num"] = pd.to_numeric(df["price_num"], errors="coerce")
df = df[df["price_num"].notna()].copy()

y = df["price_num"].copy()
X = df.drop(columns=["price_num"], errors="ignore").copy()

# Lasso前：类别压缩
def compress_topk(s: pd.Series, topk=20, other="__OTHER__", missing="__MISSING__"):
    s = s.astype("object")
    s = s.where(s.notna(), missing)
    vc = s.value_counts(dropna=False)
    kept = set(vc.head(topk).index.tolist())
    return s.where(s.isin(kept), other)

TOPK_NEIGH = 20
if "neighbourhood__lf_merge" in X.columns:
    X["neighbourhood__lf_merge"] = compress_topk(X["neighbourhood__lf_merge"], topk=TOPK_NEIGH)
for c in ["room_type","neighbourhood_group","cancellation_policy","neighbourhood__lf_merge"]:
    if c in X.columns:
        X[c] = X[c].astype(str).str.strip().str.lower()

# 导出压缩后的表格
df_table = X.copy()
df_table["price_num"] = y.values
df_table.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
print("已导出（表格版）：", OUT_TABLE)

# one-hot 展开加导出
cat_cols = [c for c in ["room_type", "neighbourhood_group", "cancellation_policy", "neighbourhood__lf_merge"] if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

Z = pre.fit_transform(X)  
feat_names = pre.get_feature_names_out()

df_onehot = pd.DataFrame(Z, columns=feat_names)
df_onehot["price_num"] = y.values

df_onehot.to_csv(OUT_ONEHOT, index=False, encoding="utf-8-sig")
print("已导出（建模版 one-hot）：", OUT_ONEHOT)
print("最终 one-hot 特征列数：", len(feat_names))


# In[11]:


from sklearn.metrics import f1_score, confusion_matrix, classification_report

def evaluate(model, X_test, y_test, name="模型"):
    y_pred = model.predict(X_test)
    print(f"\n== {name} ==")
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))


# In[1]:


# -------------------------
# 7 建模对比
# -------------------------
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
plt.rcParams["axes.unicode_minus"] = False
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"     
myfont = fm.FontProperties(fname=FONT_PATH)
DATA_PATH = "chapter6_ready_onehot51.csv"  
df = pd.read_csv(DATA_PATH)

# 构造标签（80%）
price = pd.to_numeric(df["price_num"], errors="coerce")
q80 = price.quantile(0.80)
y = (price >= q80).astype(int)

# 构造最终特征矩阵（删除 price_num ）
X = df.drop(columns=["price_num"])

# 分层划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline：Logistic Regression
clf = LogisticRegression(solver="saga", max_iter=5000)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("=== 7.3.1 Baseline：Logistic Regression ===")
print(f"样本量 N: {len(df):,} | 特征数 p: {X.shape[1]} | 阈值 q80: {q80:.0f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("混淆矩阵 (TN FP / FN TP):\n", cm)
print("\n分类报告：\n", classification_report(y_test, y_pred, digits=4, zero_division=0))

# 可视化
plt.figure(figsize=(6.6, 5.6))
ax = plt.gca()

im = ax.imshow(cm, cmap="Purples")  

ax.set_title("Baseline：Logistic Regression 混淆矩阵", fontproperties=myfont, fontsize=13)
ax.set_xlabel("预测类别", fontproperties=myfont, fontsize=11)
ax.set_ylabel("真实类别", fontproperties=myfont, fontsize=11)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["非高价(0)", "高价(1)"], fontproperties=myfont, fontsize=10)
ax.set_yticklabels(["非高价(0)", "高价(1)"], fontproperties=myfont, fontsize=10)

for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center",
            fontsize=12, color="#F4A7C5")

cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("样本数（颜色越深越多）", fontproperties=myfont, fontsize=10)

ax.text(
    0.5, -0.12,
    f"Macro-F1 = {macro_f1:.4f}（q80={q80:.0f}，测试集占比：{len(y_test)/len(y):.0%}）",
    transform=ax.transAxes,
    ha="center", va="top", fontproperties=myfont, fontsize=10, color="#C77DFF"
)

plt.tight_layout()
plt.show()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
plt.rcParams["axes.unicode_minus"] = False
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"      

myfont = fm.FontProperties(fname=FONT_PATH)

df = pd.read_csv("chapter6_ready_onehot51.csv")

# 标签（80%）
price = pd.to_numeric(df["price_num"], errors="coerce")
q80 = price.quantile(0.80)
y = (price >= q80).astype(int)

X = df.drop(columns=["price_num"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.clip(lower=-1e6, upper=1e6)
X = X.astype(np.float32)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

svm = LinearSVC(
    class_weight="balanced",
    random_state=42,
    dual=False,
    tol=1e-3,
    max_iter=2000,
    C=0.1
)
svm.fit(X_train, y_train)
print("训练完成 ")

# 评估
y_pred = svm.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("=== 7.3.2 Linear SVM（class_weight=balanced）===")
print(f"样本量 N: {len(df):,} | 特征数 p: {X.shape[1]} | 阈值 q80: {q80:.0f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("混淆矩阵 (TN FP / FN TP):\n", cm)
print("\n分类报告：\n", classification_report(y_test, y_pred, digits=4, zero_division=0))

plt.figure(figsize=(6.6, 5.6))
ax = plt.gca()
im = ax.imshow(cm, cmap="Purples")

ax.set_title("Linear SVM（balanced）混淆矩阵", fontproperties=myfont, fontsize=13)
ax.set_xlabel("预测类别", fontproperties=myfont, fontsize=11)
ax.set_ylabel("真实类别", fontproperties=myfont, fontsize=11)

ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["非高价(0)", "高价(1)"], fontproperties=myfont, fontsize=10)
ax.set_yticklabels(["非高价(0)", "高价(1)"], fontproperties=myfont, fontsize=10)

for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12, color="#F4A7C5")

cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("样本数（颜色越深越多）", fontproperties=myfont, fontsize=10)

ax.text(
    0.5, -0.12,
    f"Macro-F1 = {macro_f1:.4f}（q80={q80:.0f}，测试集占比：{len(y_test)/len(y):.0%}）",
    transform=ax.transAxes, ha="center", va="top",
    fontproperties=myfont, fontsize=10, color="#C77DFF"
)

plt.tight_layout()
plt.show()


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
plt.rcParams["axes.unicode_minus"] = False
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"      

myfont = fm.FontProperties(fname=FONT_PATH)

DATA_PATH = "chapter6_ready_onehot51.csv"  # 改成你的路径
df = pd.read_csv(DATA_PATH)
price = pd.to_numeric(df["price_num"], errors="coerce")
q80 = price.quantile(0.80)  # 一般为 971
y = (price >= q80).astype(int)
X = df.drop(columns=["price_num"]).copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
arr = X.to_numpy()
mask_bad = ~np.isfinite(arr) 
arr = np.clip(arr, -1e6, 1e6)
arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
X = pd.DataFrame(arr, columns=X.columns).astype(np.float64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("开始训练 RandomForest...")

rf = RandomForestClassifier(
    n_estimators=300,         
    max_depth=None,            
    min_samples_leaf=3,        
    n_jobs=-1,                 
    random_state=42,
    class_weight="balanced_subsample"
)
rf.fit(X_train, y_train)

print("训练完成 ")

y_pred = rf.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("=== 7.3.3 RandomForest（balanced_subsample）===")
print(f"样本量 N: {len(df):,} | 特征数 p: {X.shape[1]} | 阈值 q80: {q80:.0f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("混淆矩阵 (TN FP / FN TP):\n", cm)
print("\n分类报告：\n", classification_report(y_test, y_pred, digits=4, zero_division=0))

plt.figure(figsize=(6.6, 5.6))
ax = plt.gca()

im = ax.imshow(cm, cmap="Purples")
ax.set_title("RandomForest（balanced_subsample）混淆矩阵", fontproperties=myfont, fontsize=13)
ax.set_xlabel("预测类别", fontproperties=myfont, fontsize=11)
ax.set_ylabel("真实类别", fontproperties=myfont, fontsize=11)

ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["非高价(0)", "高价(1)"], fontproperties=myfont, fontsize=10)
ax.set_yticklabels(["非高价(0)", "高价(1)"], fontproperties=myfont, fontsize=10)

for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center",
            fontsize=12, color="#F4A7C5")

cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("样本数（颜色越深越多）", fontproperties=myfont, fontsize=10)

ax.text(
    0.5, -0.12,
    f"Macro-F1 = {macro_f1:.4f}（q80={q80:.0f}，测试集占比：{len(y_test)/len(y):.0%}）",
    transform=ax.transAxes, ha="center", va="top",
    fontproperties=myfont, fontsize=10, color="#C77DFF"
)

plt.tight_layout()
plt.show()


# In[ ]:




