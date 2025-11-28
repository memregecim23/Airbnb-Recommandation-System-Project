import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from datetime import date
import ast
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("csv/Listings.csv", encoding='latin1', low_memory=False)
df_r = pd.read_csv("csv/Reviews.csv", encoding='latin1', low_memory=False)

# Tarih ifade eden değişkenlerin tipinin date'e çevrilmesi
review_date = [col for col in df_r.columns if "date" in col]
df_r[review_date] = df_r[review_date].apply(pd.to_datetime)
df_r[df_r['date'].dt.year >= 2017]["listing_id"].nunique()

df["host_since"] = pd.to_datetime(df["host_since"])

# 'listing_id' sütununa göre veri setini birleştirme
df_airbnb = pd.merge(df_r, df, on='listing_id', how='left')

# 2017'den eski verileri silme
df_airbnb.drop(df_airbnb[df_airbnb['date'].dt.year < 2017].index, inplace=True)

# district sütununun kaldırılması
df_airbnb.drop(columns=["district"], inplace=True)

###############################################################################################################
# EKSIK DEGER PROBLEMLERİNİN COZULMESİ:

# SUDE NİN PART
# bedrooms eksik değer problemi çözümü:

# "name" boş olan ilanlar veri setinden kaldırıldı. (175 gözlem birimi)
df_airbnb.dropna(subset=["name"], inplace=True)

# "studio" kelimesini içeren ilanların seçilmesi
df_airbnb["name_has_studio"] = df_airbnb["name"].str.contains("studio", case=False, na=False)

# ilanda studio yazıp bedroom sayısı 1'den büyük olanlar alınıp çelişkili olarak atandı
df_studio_conflict = df_airbnb[(df_airbnb["name_has_studio"]) & (df_airbnb["bedrooms"] > 1)]

# studio olup değeri 0 olan var mı kontrol:
df_studio_zero = df_airbnb[(df_airbnb["name_has_studio"]) & (df_airbnb["bedrooms"] == 0)]
print(f"Studio yazıp bedrooms = 0 olan ilan sayısı: {len(df_studio_zero)}")

# studio olup 1 den fazla olanları ayrı alalım:
df_airbnb["studio_bedroom_conflict"] = (
    (df_airbnb["name_has_studio"]) & (df_airbnb["bedrooms"] > 1)
).astype(int)

# Eksik (NaN) bedroom değeri olup, 'studio' geçen ilanlara 1 ata
df_airbnb.loc[
    (df_airbnb["bedrooms"].isna()) & (df_airbnb["name_has_studio"]),
    "bedrooms"
] = 1

# Kalan diğer eksik bedroom değerlerini medyan ile doldur
median_bedroom = df_airbnb["bedrooms"].median()
df_airbnb["bedrooms"].fillna(median_bedroom, inplace=True)

##############################################################################################################

# host eksiklikleri ve amenities one hot encoding bölümü

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def one_hot_encode(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first = drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df_airbnb)

num_cols = [col for col in num_cols if col not in ("listing_id","host_id","host_since","latitude","longitude")]

grab_col_names(df_airbnb)

# boş değerli sütunlar
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

# host_total_listings_count ortalama ile doldurdum
df_airbnb["host_total_listings_count"] = df_airbnb["host_total_listings_count"].fillna(df_airbnb["host_total_listings_count"].mean())
df_airbnb["host_total_listings_count"].info

# host_has_profile_pic mod ile doldurdum
df_airbnb["host_has_profile_pic"] = df_airbnb["host_has_profile_pic"].fillna(df_airbnb["host_has_profile_pic"].mode()[0])
df_airbnb["host_has_profile_pic"].info

# bu durum host_is_superhost durumunu etkileyeceğinden string değerle doldurdum
df_airbnb["host_identity_verified"] = df_airbnb["host_identity_verified"].fillna("unknown")
df_airbnb["host_identity_verified"].info

# string ifade ile doldurdum
df_airbnb["host_since"] = df_airbnb["host_since"].fillna("long_ago")
df_airbnb["host_since"].info

# string ifade ile doldurdum
df_airbnb["host_location"] = df_airbnb["host_location"].fillna("in_the_world")
df_airbnb["host_location"].info

# host_is_superhost olma durumu
# Eğer host_is_superhost değeri eksikse ve cevap oranı %90’dan düşükse, bu ev sahibi muhtemelen "süper ev sahibi" değildir
df_airbnb["host_is_superhost"] = df_airbnb.apply(
    lambda x: "f" if pd.isna(x["host_is_superhost"]) and x["host_response_rate"] < 0.90
    else ("t" if pd.isna(x["host_is_superhost"]) else x["host_is_superhost"]),
    axis=1)

# Sadece her listing_id için ilk satırı al
df_unique = df_airbnb.dropna(subset=["amenities"]).drop_duplicates(subset="listing_id")

amenity_counter = Counter()

for row in df_unique["amenities"]:
    try:
        amenities = ast.literal_eval(row)
        amenity_counter.update(amenities)
    except Exception as e:
        print("Hata:", e)
        continue

# En çok geçen ilk 20 özellik
top_20_amenities = [amenity for amenity, count in amenity_counter.most_common(20)]
print("Top 20 amenities:", top_20_amenities)

# Her bir top amenity için binary sütun oluşturma
for amenity in top_20_amenities:
    df_unique[f"has_{amenity.replace(' ', '_')}"] = df_unique["amenities"].apply(
        lambda x: 1 if amenity in ast.literal_eval(x) else 0
    )

cols = [col for col in df_unique.columns if col.startswith("has_")]
df_airbnb = df_airbnb.merge(
    df_unique[["listing_id"] + cols],
    on="listing_id",
    how="left"
)

df_airbnb = one_hot_encode(df_airbnb, categorical_cols=["host_is_superhost", "host_has_profile_pic", "host_identity_verified"], drop_first=True)
df_airbnb["host_has_profile_pic_t"] = df_airbnb["host_has_profile_pic_t"].astype(int)
df_airbnb["host_is_superhost_t"] = df_airbnb["host_is_superhost_t"].astype(int)
df_airbnb["host_identity_verified_t"] = df_airbnb["host_identity_verified_t"].astype(int)

########################################################################################################

# skor eksik değerlerini tahmine dayalı doldurma:
"""
review_cols = [col for col in df_airbnb.columns if col.startswith("review_scores_")]

df_review = df_airbnb[review_cols].copy()

scaler = MinMaxScaler()
df_review = pd.DataFrame(scaler.fit_transform(df_review), columns=df_review.columns)

imputer = KNNImputer(n_neighbors=5)
df_review = pd.DataFrame(imputer.fit_transform(df_review), columns=df_review.columns)
"""
# ortalama ile dolduruldu
review_cols = [col for col in df_airbnb.columns if col.startswith("review_scores_")]
df_airbnb[review_cols] = df_airbnb[review_cols].fillna(df_airbnb[review_cols].mean())

######## Price #######
city_to_country = {
    "Paris": "France",
    "New York": "USA",
    "Bangkok": "Thailand",
    "Rio de Janeiro": "Brazil",
    "Sydney": "Australia",
    "Istanbul": "Turkiye",
    "Rome": "Italy",
    "Hong Kong": "China",
    "Mexico City": "Mexico",
    "Cape Town": "South Africa"
}

df_airbnb["country"] = df_airbnb["city"].map(city_to_country)

# Ortalama kurlar
exchange_rates = {
    "Mexico": 19.94,  # MXN
    "South Africa": 14.47,  # ZAR
    "Thailand": 31.94,  # THB
    "China": 6.80,  # CNY
    "Brazil": 4.03,  # BRL
    "Turkiye": 5.37,  # TRY
    "Australia": 1.0,
    "France": 1.0,
    "Italy": 1.0,
    "USA": 1.0
}
df_airbnb["price"] = df_airbnb.apply(
    lambda row: row["price"] / exchange_rates.get(row["country"], 1.0), axis=1)

df_airbnb.groupby("country")["price"].describe([0.0004, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9996])

# Min baskılama
for country in df_airbnb["country"].unique():
    threshold = max(df_airbnb[df_airbnb["country"] == country]["price"].quantile(0.15), 20)
    df_airbnb.loc[(df_airbnb["country"] == country) & (df_airbnb["price"] < threshold), "price"] = threshold

# Max baskılama
for country in df_airbnb["country"].unique():
    threshold = df_airbnb[df_airbnb["country"] == country]["price"].quantile(0.9985)
    df_airbnb.loc[(df_airbnb["country"] == country) & (df_airbnb["price"] > threshold), "price"] = threshold

df_airbnb.groupby("country")["price"].describe([0.01, 0.25, 0.4, 0.5, 0.6, 0.75, 0.99])

# KONTROL BLOĞU
print("Final shape:", df_airbnb.shape)
na_ratio = (df_airbnb.isnull().mean() * 100).sort_values(ascending=False).head(10)
print("En fazla NA oranına sahip ilk 10 sütun (%):")
print(na_ratio)
