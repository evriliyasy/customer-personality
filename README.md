# Laporan Proyek Machine Learning

# Nama : Evriliya Syah Utami

# Nim : 211351051

# Kelas : TIF Pagi A

# Domain Proyek

proyek ini berfokus pada analisis kepribadian pelanggan untuk membantu perusahaan memahami para pelanggan. Dalam konteks ini, kita akan menggunakan data pelanggan untuk mengidentifikasi kebutuhan, dan preferensi pelanggan. Dengan memahami personalitas pelanggan, perusahaan dapat meningkatkan strategi pemasaran. 

 # Business Understanding

Untuk mengidentifikasi segmen pelanggan yang berbeda berdasarkan kebutuhan pelanggan dan meningkatkan penjualan.

# Problem Statements

* Tantangan saya dalam proyek ini adalah,  mengelompokkan pelanggan ke dalam segmen-segmen berdasarkan variabel-variabel tertentu seperti kebutuhan pelanggan, frekuensi kunjungan, dan jumlah pembelian.

# Goals

* Tujuan utama adalah untuk mengidentifikasi segmen pelanggan berdasarkan perilaku pembelian, preferensi, dan karakteristik lainnya.
* Menghasilkan pemahaman mendalam tentang preferensi dan kebiasaan belanja pelanggan untuk membantu perusahaan mengoptimalkan strategi pemasaran dan penjualan.

# Solution Statements

Solusi untuk proyek ini adalah mengumpulkan dan mengelompokkan pelanggan ke dalam segmen-segmen berdasarkan kesamaan perilaku dan karakteristik pembelian. Data ini akan digunakan untuk melatih model machine learning yang dapat mengelompokkan analisis kepribadian pelanggan untuk strategi pemasaran yang lebih efektif.

# Data Understanding

Mengelompokkan analisis kepribadian pelanggan untuk membuat streategi pemasaran sesuai dengan kebutuhan dan preferensi khusus dari setiap segmen pelanggan.
Proyek ini berguna untuk menganalisis apa yang dibutuhkan para pelanggan.
inilah datasets yang saya ambil (https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering/notebook).

# Variabel-variabel Daftar Analisis Kepribadian Pelanggan 

* Beras Termurah = Beras dengan harga terjangkau dengan type (int64)
* Income = dengan type (float64)
* Kidhome = dengan type (int64)
* Teenhome = dengan type (int64)
* Recency = dengan type (int64)
* MntWines = dengan type (int64)
* MntFruits = dengan type (int64)
* MeatProducts = dengan type (int64)
* FishProducts = dengan type (int64)
* SweetProducts = dengan type (int64)
* GoldProducts = dengan type (int64)
* NumDealsPurchases = dengan type (int64)
* NumCatalogPurchases = dengan type (int64)
* NumStorePurchases = dengan type (int64)
* Cost = dengan type (int64)
* Revenue = dengan type (int64)

# Data Preparation

Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama Daftar Analisis Kepribadian Pelanggan, jika anda tertarik dengan datasetnya, anda bisa click link diatas.

# Data Discovery 

Untuk bagian ini, kita akan menggunakan teknik EDA.
Pertama kita mengimport semua library yang dibutuhkan,

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import pickle
```

Karena kita menggunakan google colab untuk mengerjakannya maka kita akan import files juga,

```python
from google.colab import files
```

Lalu mengupload token kaggle agar nanti bisa mendownload sebuah dataset dari kaggle melalui google colab

```python
files.upload()
```

Setelah mengupload filenya, maka kita akan lanjut dengan membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload tadi

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Done, lalu mari kita download datasetsnya

```ptrhon
!kaggle datasets download -d imakash3011/customer-personality-analysis 
```

Selanjutnya kita harus extract file yang tadi telah didownload 

```python
!mkdir customer-personality-analysis
!unzip customer-personality-analysis.zip -d customer-personality-analysis
!ls customer-personality-analysis
```

Mari lanjut dengan memasukkan file csv yang telah diextract pada sebuah variable, dan melihat 5 data paling atas dari datasetsnya

```python
df = pd.read_csvdf = pd.read_csv('customer-personality-analysis/marketing_campaign.csv',  sep="\t")
```

Untuk melihat beberapa baris pertama dari sebuah DataFrame.

```python
df.head()
```
Untuk melihat mengenai type data dari masing masing kolom kita bisa menggunakan property info,

```python
df.info()
```
menghasilkan statistik deskriptif tentang DataFrame, seperti rata-rata, median, kuartil, dan lainnya, untuk setiap kolom numerik dalam DataFrame

```python
df.describe()
```

Untuk melihat beberapa baris terakhir dari sebuah DataFrame.

```python
df.isnull().sum()
```

```python
sns.heatmap(df.isnull())
```
# EDA

Mari kita lanjut dengan visualisai data kita, dan akan munsul atribut yang numerik atau integer

```python
df.hist(figsize=(20, 15))
plt.show()
```

```python
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(),annot=True)
plt.title("heatmap korelasi (numerik)")
plt.show()
```
Membuat beberapa plot atau garfik
Pertama membuat grup Wines dan Fruits

```python
models = df.groupby('MntWines').count()[['MntFruits']].sort_values(by='MntFruits',ascending=True).reset_index()
models = models.rename(columns={'MntFruits' : 'patient'})
```
Lalu membuat tipikal grafik dalam bentuk barplot

```python
pig = plt.figure(figsize=(15,5))
sns.barplot(x=models['MntWines'], y=models['patient'], color='royalblue')
plt.xticks(rotation=60)
plt.show()
```
Membuat beberapa plot atau garfik
Kedua membuat grup Marital Status dan Wines

```python
engine = df.groupby('Marital_Status').count()[['MntWines']].sort_values(by='MntWines',ascending=True).reset_index()
engine = engine.rename(columns={'MntWines': 'patient'})
```
Lalu membuat tipikal grafik dalam bentuk barplot

```python
eplt.figure(figsize=(15,5))
sns.barplot(x=engine['Marital_Status'], y=engine['patient'], color='royalblue')
```

```python
x = df.drop(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Dt_Customer', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response'], axis=1)
```

```python
plt.title('Income VS Year_Birth')
plt.scatter(x['Income'], x['Recency'])
plt.xlabel('Income')
plt.ylabel('Year_Birth')
plt.grid()
plt.show()
```
Untuk menampilkan data dari satu kolom tertentu,

```python
print(f"Columns in DataFrame: {df.columns}")
```
Untuk Membuat plot Income terhadap fitur lainnya,

```python
data_compare = df[['Income', 'MntMeatProducts', 'MntFishProducts', 'MntGoldProds']]
# Plot Income terhadap fitur lainnya
for feature in ['MntMeatProducts', 'MntFishProducts', 'MntGoldProds']:
    plt.scatter(data_compare['Income'], data_compare[feature], label=feature, s=10)

plt.xlabel('Income')
plt.ylabel('Feature Values')
plt.title('Comparison of Income with Other Features')
plt.legend()
plt.show()
```

# Data Preparation 

# Merubah Nama Columns

```python
df.rename(index=str, columns={
    'MntMeatProducts' : 'MeatProducts',
    'MntFishProducts' : 'FishProducts',
    'MntSweetProducts' : 'SweetProducts',
    'MntGoldProds' : 'GoldProducts',
    'Z_CostContact' : 'Cost',
    'Z_Revenue' : 'Revenue',

}, inplace=True)
```

# Menghapus Columns yang tidak digunakan 

```python
x = df.drop(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Dt_Customer', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response'], axis=1)
```

Menampilkan x 

```python
print(x)
```
Menampilkan type data,

```python
print(df.dtypes)
```
Untuk membuat DataFrame baru,

```python
x_cleaned = x.dropna()
```

Mari kita lanjut ke modeling

# Modeling

Langkah pertama kita melakukan seleksi fitur karena tidak semua antribut yang ada didataset kita pakai

Memilih fitur yang ada di dataset dan penamaan atau huruf harus sama seperti di dataset supaya terpanggil serta menentukan featurs dan labels

# Menentukan Jumlah Cluster Denagn Elbow

```python
imputer = SimpleImputer(strategy='mean')
x_imputed = imputer.fit_transform(x)
```

```python
clusters= []
for i in range(1, 11):
    km =KMeans(n_clusters=i).fit(x_imputed)
    clusters.append(km.inertia_)


fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

#Panah elbow
ax.annotate('possible elbow point', xy=(8, 4.5), xytext=(8, 2.5), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('possible elbow point', xy=(4, 4.5), xytext=(4, 2.5), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
```

# Visualisasi hasil modeling

# Proses Clustering

```python
n_clust = 4
kmean = KMeans(n_clusters=n_clust).fit(x_imputed)
x['labels'] = kmean.labels_
```
Menampilkan plot 

```python
plt.figure(figsize=(10, 8))
sns.scatterplot(x=x['SweetProducts'], y=x['GoldProducts'], hue=x['labels'], palette=sns.color_palette('hls', n_colors=n_clust))

for label in x['labels']:
    plt.annotate(label,
                 (x[x['labels'] == label]['SweetProducts'].mean(),
                  x[x['labels'] == label]['GoldProducts'].mean()),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')
```
Menampilkan x

```python
print(x)
```
Menampilkan cluster 
```python
cluster_means = x.groupby('labels').mean()
print(cluster_means)
```
Sekarang modelnya sudah selesai, mari kita export sebagai excel agar nanti bisa kita gunakan pada project web streamlit kita.

```python
import pickle

x.to_excel("output_cluster.xlsx")
```

# Evaluation

# Impor pustaka yang diperlukan

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Menggunakan model K-Means

kmeans_model = KMeans(n_clusters=K)  # K adalah jumlah klaster yang diinginkan
kmeans_model.fit(data)

# Evaluasi menggunakan Inertia
inertia = kmeans_model.inertia_
print(f"Inertia: {inertia}")

# Deployment
