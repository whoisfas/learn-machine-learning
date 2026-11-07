---
{
  "id": "p1-pandas",
  "title": "Pandas: DataFrames, Cleaning, Groupby",
  "phase": 1,
  "difficulty": "beginner",
  "estimated_hours": 14,
  "objectives": [
    "Load, inspect, and clean tabular data with pandas",
    "Use boolean indexing, `.loc`, `.iloc`, and `.query`",
    "Aggregate with groupby and pivot_table",
    "Handle missing values, duplicates, and dtypes correctly"
  ],
  "concepts": [
    "Series vs DataFrame",
    "dtypes: int64, float64, object, category, datetime64",
    "Indexing: label (.loc) vs position (.iloc)",
    "Boolean masks and chained assignment pitfalls (SettingWithCopyWarning)",
    "Missing data: NaN, isna, fillna, dropna, interpolation",
    "groupby: agg, transform, apply",
    "merge/join/concat",
    "applymap/apply, vectorized string ops (.str)",
    "Categorical dtype for memory + speed"
  ],
  "formulas": [],
  "references": [
    "Pandas user guide: https://pandas.pydata.org/docs/user_guide/",
    "10 minutes to pandas"
  ]
}
---

## Penjelasan

Pandas menyediakan `DataFrame` (tabel 2D) dan `Series` (kolom 1D) di atas NumPy. Hampir semua dataset ML berbentuk tabel, jadi pandas adalah alat utama untuk EDA dan preprocessing.

**Struktur.** DataFrame punya index (baris, default RangeIndex) dan columns. `df.shape`, `df.dtypes`, `df.head()`, `df.info()`, `df.describe()` untuk inspeksi cepat. `df[col]` -> Series; `df[[col1,col2]]` -> DataFrame.

**Indexing.** `.loc` berbasis label: `df.loc[df.age > 30, ['name','age']]`. `.iloc` berbasis posisi integer: `df.iloc[0:5, 0:2]`. `.at`/`.iat` untuk scalar. `.query('age > 30 & city == "A"')` alternatif ekspresif.

**Missing data.** `NaN` (float), `None` (object), `pd.NA` (nullable). Deteksi: `df.isna().sum()`. Strategi: `dropna`, `fillna(value)`, `fillna(method='ffill')` (deprecated -> `ffill()`), `interpolate()`. Keputusan imputation harus dipertimbangkan: mean imputation mengurangi varians; tambahkan indikator missing jika missingness informatif.

**Groupby.** `df.groupby('city')['sales'].agg(['mean','count'])`. `transform` mengembalikan hasil seukuran indeks asli (untuk normalisasi per grup). `apply` fleksibel tapi lambat. `pivot_table` untuk cross-tabulation dengan agregasi.

**Merge.** `pd.merge(df1, df2, on='id', how='left')` — analog SQL JOIN. `concat` untuk menumpuk baris/kolom. Hati-hati duplikasi baris akibat many-to-many.

**Dtypes & memory.** Ubah ke `category` untuk kolom string berulang (hemat memori signifikan). `pd.to_datetime` untuk tanggal. `pd.to_numeric(errors='coerce')` untuk parsing angka dari string kotor.

## Konsep Penting

1. **SettingWithCopyWarning** — `df[df.x>0]['y'] = 1` mungkin tidak mengubah df asli. Pakai `.loc` atau `.copy()`.
2. **`inplace=True` deprecated** — gunakan `df = df.dropna()` daripada `df.dropna(inplace=True)`.
3. **NaN != NaN** — `df == np.nan` tidak bekerja; pakai `isna()`.
4. **Groupby + transform** — `df['pct'] = df.groupby('g')['v'].transform(lambda s: s/s.sum())` memberi persentase per grup per baris.
5. **`apply` lambat** — untuk operasi baris, coba `vectorize` atau ekspresi vectorized dulu.

## Formula

- Memory: category dtype ~ 1 byte/kategori vs object ~ 50+ byte/string.
- groupby agg = split-apply-combine pattern.

## Contoh Kasus

Analisis dataset e-commerce: merge orders+customers+products (3 tabel), handle missing customer region, groupby produk menghitung revenue/quantity, pivot_table untuk revenue per kategori per bulan, deteksi outlier dengan IQR per grup.

## Implementasi

```python
import pandas as pd
import numpy as np

df = pd.read_csv("sales.csv", parse_dates=["date"])
df["month"] = df["date"].dt.to_period("M")
df["revenue"] = df["qty"] * df["price"]

# per-customer share of revenue
df["rev_share"] = df.groupby("customer_id")["revenue"].transform(
    lambda s: s / s.sum())

# monthly pivot
pivot = df.pivot_table(index="month", columns="category",
                       values="revenue", aggfunc="sum", fill_value=0)

# safe filtering + assignment
df.loc[df["qty"] < 0, "qty"] = np.nan  # returns or errors flagged
df = df.dropna(subset=["qty"])
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Given df with columns A=[1,2,3], B=[4,5,6], what does `df.loc[df.A > 1, 'B'].sum()` return?",
    "options": ["11", "9", "15", "Error"],
    "answer": 1,
    "explanation": "Rows where A>1 are rows 2,3 -> B values 5+6=11? No: B=[4,5,6], rows 1,2 -> 5+6=11. Correct: 11.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "coding",
    "question": "Implement `group_mean(df, group_col, value_col)` returning a Series of per-group means aligned to original index (like transform).",
    "starter_code": "import pandas as pd\n\ndef group_mean(df, group_col, value_col):\n    pass\n",
    "function": "group_mean",
    "tests": [
      {"call": "group_mean(pd.DataFrame({'g':['a','a','b'],'v':[1,3,10]}),'g','v').tolist()", "expected": [2.0, 2.0, 10.0]}
    ],
    "hints": ["df.groupby(group_col)[value_col].transform('mean')"],
    "explanation": "transform broadcasts group stats back to rows.",
    "points": 2
  },
  {
    "id": "ex3",
    "type": "coding",
    "question": "Implement `fill_missing_with_median(df)` returning a copy where NaNs in numeric columns are filled with the column median.",
    "starter_code": "import pandas as pd\nimport numpy as np\n\ndef fill_missing_with_median(df):\n    pass\n",
    "function": "fill_missing_with_median",
    "tests": [
      {"call": "fill_missing_with_median(pd.DataFrame({'a':[1.0,np.nan,3.0]})).values.tolist()", "expected": [[1.0],[2.0],[3.0]]}
    ],
    "hints": ["df.copy() first", "df.median(numeric_only=True) then fillna"],
    "explanation": "Median imputation preserves central tendency better than mean for skewed data.",
    "points": 2
  },
  {
    "id": "ex4",
    "type": "open",
    "question": "A column 'income' has 30% missing values concentrated in one demographic group. Discuss (a) why dropping rows is dangerous here, (b) two imputation strategies and their biases, (c) when to add a missingness indicator feature.",
    "rubric": [
      "Identifies selection bias / MNAR problem",
      "Compares mean vs median vs model-based imputation",
      "Mentions missingness indicator for informative missingness"
    ],
    "points": 2
  }
]
```

## Referensi

- Pandas user guide (indexing, groupby, reshaping)
- "Python for Data Analysis" ch. 5-10
- pandas.pydata.org — cookbook
