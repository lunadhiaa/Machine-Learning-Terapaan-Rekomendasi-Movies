# Laporan Proyek Machine Learning - Lulu Nadhiatun Anisa

## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah meninjau **Rekomenadisi Film **  

## Project Overview
![Film](https://github.com/user-attachments/assets/79169bc0-4e92-4d67-9e9d-d8c40e0462c6)

Di era digital, informasi yang tersedia  dan dapat diakses melalui internet kapanpun dan dimanapun setiap harinya.
Dalam dunia bisnis modern, informasi telah menjadi kebutuhan utama, dengan penggunaan basis data sebagai kunci untuk menghasilkan solusi bisnis yang inovatif. 
Banyak company kini memanfaatkan data untuk memahami preferensi pelanggan, yang pada akhirnya mendorong perkembangan sistem rekomendasi. 
Sistem ini menjadi alat penting untuk meningkatkan pengalaman pengguna dengan menyediakan rekomendasi yang relevan dan sesuai dengan kebutuhan mereka.[[1](https://www.antaranews.com/berita/1776245/layanan-streaming-tantangan-dan-peluang-perfilman-indonesia#google_vignette)]

Maraknya layanan streaming seperti Netflix, Disney Plus, WeTV, dan platform serupa telah menciptakan tren baru dalam pola menonton film, seperti binge-watching dan hopping. 
Tren ini mengacu pada kebiasaan menonton film secara berurutan tanpa jeda.Dengan semakin banyaknya konten dan pilihan genre yang tersedia, 
penyedia layanan streaming dihadapkan pada tantangan untuk memberikan rekomendasi yang relevan agar dapat meningkatkan kepuasan serta loyalitas pengguna.[[2](https://kumparan.com/millennial/mengenal-istilah-binge-watching-dan-hopping-yang-jadi-tren-terbaru-nonton-film-1wREvjSoXeB)]

Dari permasalhan tersebut, proyek ini bertujuan untuk mengembangkan model machine learning yang mendukung sistem rekomendasi film.
Model ini dirancang dengan memanfaatkan data preferensi pengguna berdasarkan histori kesukaan mereka sebelumnya, serta penilaian atau rating dari berbagai pengguna terhadap film tertentu. 
Dengan pendekatan ini, sistem rekomendasi yang dihasilkan diharapkan mampu memberikan rekomendasi yang lebih personal dan akurat, sehingga membantu pengguna menemukan film yang sesuai dengan minat mereka.

## Business Understanding
Pengembangan model sistem rekomendasi film memiliki berbagai manfaat, baik bagi pengguna maupun penyedia layanan seperti efisiensi pencarian, ekplorasi kontenbaru,
peningkatan loyalisat pengguna, optimasi pendapatan dan lainnya.

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan, berikut adalah rincian masalah yang dapat diatasi melalui proyek ini:
- Bagaimana cara membuat sebuah sistem rekomendasi film yang disukai pengguna lain dapat direkomendasikan kepada pengguna lainnya juga?
- Bagaimana cara menyajikan sistem rekomendasi film berdasarkan ratings film tertinggi?

### Goals
Mampu merancang dan mengembangkan sistem rekomendasi yang secara akurat menyarankan konten berdasarkan analisis mendalam terhadap 
rating yang diberikan pengguna serta aktivitas mereka di masa lalu. 

### Solution approach

Solusi yang saya kembangkan menggunakan dua algoritma machine learning untuk mendukung sistem rekomendasi, yaitu:

- **Content Based Filtering** adalah salah satu metode dalam sistem rekomendasi yang digunakan untuk memberikan rekomendasi kepada pengguna berdasarkan karakteristik atau atribut dari item yang telah mereka interaksikan atau sukai sebelumnya.
- **Collaborative Filtering**. adalah teknik yang digunakan dalam sistem rekomendasi untuk memberikan rekomendasi berdasarkan preferensi atau interaksi pengguna lain. Pendekatan ini tidak bergantung pada atribut atau konten dari item itu sendiri (seperti dalam Content-Based Filtering).
  
Algoritma *Content Based Filtering* digunakan untuk merekemondesikan movie berdasarkan aktivitas pengguna pada masa lalu, sedangkan algoritma *Collabarative Filltering* digunakan untuk merekomendasikan movie berdasarkan ratings yang paling tinggi.

## Data Understanding
**Informasi Datasets**

| Jenis | Keterangan |
| ------ | ------ |
| Title | Movie Recommendation Data |
| Source | [Kaggle](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data) |
| Maintainer | [Rohan Sharma⚡](https://www.kaggle.com/rohan4050) |
| License | Unknown |
| Visibility | Publik |
| Tags | Movie and TV Show, Recommender System |
| Usability | 7.94 |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah perusahaan pertanian Amerika, yang disediakan secara publik di kaggle dengan nama datasets yaitu: _Apple Quality_

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| 0.0 | -3.970049 |-2.512336 | 5.346330 |-1.012009 | 1.844900 |0.329840	| -0.491590483  |good |
| 1.0 | -1.195217 |-2.839257 | 3.664059 |1.588232 | 0.853286 | 0.867530 | -0.722809367  |good |
| 2.0 | -0.292024 |	-1.351282 | -1.738429 | -0.342616 | 2.838636 |-0.038033	| 2.621636473  |bad |
| 3.0 | -0.657196 |-2.271627 | 1.324874 |-0.097875 | 3.637970 |-3.413761	| 0.790723217  |good |
| 4.0 | 1.364217 |-1.296612 | -0.384658 | -0.553006 | 3.030874 | -1.303849	| 0.501984036  |good |

Tabel 1. EDA Deskripsi Variabel_

Dilihat dari _Tabel 2. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 4001 sample dengan 9 fitur.
- Dataset memiliki 7 fitur bertipe float64 dan 2 fitur bertipe object.
- Terdapat 1 missing value dalam dataset.

### Variabel-variabel dataset adalah sebagai berikut:
- `A_id` : Identifikasi unik untuk setiap buah.
- `Size` : Ukuran buah.
- `Weight` : Berat buah.
- `Sweetness` : Tingkat kemanisan buah.
- `Crunchiness` : Tekstur yang menunjukkan kerenyahan buah.
- `Juiciness` : Tingkat kesegaran buah.
- `Ripeness` : Tahap kematangan buah.
- `Acidity` : Tingkat keasaman buah.
- `Quality` : Kualitas buah secara keseluruhan, baik atau buruk.

Dari ke 9 fitur dapat dilihat bahwa fitur `A_id` tidak mempengaruhi kualitas buah hingga akan di hapus.

### EDA - Univariate Analysis

![univariate analysis 1a](https://github.com/user-attachments/assets/e9330311-aac9-46f8-8080-a69ea2f4872b)

Gambar 1a. Analisis Univariat (Data Kategori Apel) 

![univariate analysis 1b](https://github.com/user-attachments/assets/e9543576-081f-447e-ab7d-d4d4970b3045)

Gambar 1b. Analisis Univariat (Data Numerik) 

 Berdasarkan _Gambar 1a_ , dapat dilihat bahwa distribusi data katagorik _Quality_ yang terdiri dari _good_ dan _bad_ kualitas apel, yang mana nilai data **bad** terdiri dari `1928` dan **good** terdiri dari `1862`, yang mana menunjukan perbandingan data yang tidak terlalu jauh. Pada _Gambar 1b,_ untuk data numerik memiliki karakteristik, yaitu:
  - Dilihat dari distribusi data numerik _Size_, ukuran rata-rata buah berkisar dari -2 sampai 2, dan memiliki nilai rata-rata _Mean_ adalah -0.51.
  - Rata-rata berat apel bernilai -0.99 dan nilai _max_ berat apel adalah 3.08.
  - Rata-rata tingkat kemanisan apel -0.48.
  - Tekstur kerenyahan apel berkisar dari 0 hingga 2 yang mana nilai ini menunjukan rata-rata apel itu renyah.
  - Tingkat kesegaran buah dan Kematangan buat berada pada nilai 0.50 dan 0.53.
  - Rata-rata tingkat keasaman buah bernilai 0.06.

 Nilai-nilai ini menunjukkan bahwa data  telah dinormalisasi dengan cara _z-score normalization_ . _z-score normalization_  mengubah data dengan cara:
 - Mengurangi rata-rata (mean) dari setiap data point.
 - Membagi hasil pengurangan tersebut dengan standar deviasi data.
 
Pada kasus ini, rata-rata (mean) data "Size" adalah -0.51 dan standar deviasi data "Size" tidak diketahui. Namun, dengan nilai minimum -2 dan maksimum 2, dapat diasumsikan bahwa data "Size" telah diubah skalanya sehingga memiliki mean 0 dan standar deviasi 1. Data numerik lainnya, seperti _"Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", dan "Acidity"_, juga telah dinormalisasi dengan cara yang sama.


### EDA - Multivariate Analysis

![multi1](https://github.com/user-attachments/assets/d926d5c4-8dfd-4a11-acf8-5082ebdf9340)

Gambar 2a. Analisis Multivariat

![multi2](https://github.com/user-attachments/assets/f615c6c9-3e5d-4f09-9be9-42bb4c1184a4)

Gambar 2b. Analisis Karakteristik apel

![multi3](https://github.com/user-attachments/assets/b3bd07ae-426a-4984-8776-42718fc7c2f3)

Gambar 2c. Analisis Matriks Korelasi

Pada _Gambar 2a. Analisis Multivariat_, dengan menggunakan fungsi _pairplot_ dari _library seaborn_, tampak terlihat relasi pasangan dalam dataset menunjukan pola acak. Pada pola sebaran data grafik pairplot, terterlihat bahwa _Size_ dan _Sweetness_ memiliki korelasi negatif menurun, yang mana semakin kecil ukuran buah rasa nya akan semakin manis.

Pada _Gambar 2b. Analisis Karakteristik_ menggunakan box plot dengan membandingkan distribusi berbagai karakteristik apel antara apel yang diklasifikasikan dalam kualitas *Good* dan *Bad*. Sn Dengan sebaran data untuk berbagai fitur seperti *size, weight, sweetness, crunchiness, juiciness, ripeness, and acidity*.

Pada _Gambar 2c. Analisis Matriks Korelasi_, merupakan _Correlation Matrix_ menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur _Juiciness_ memiliki skor korelasi yang cukup besar `0.24` dengan fitur target _Acidity_ .

## Data Preparation
Pada proses _Data Preparation_ dilakukan kegiatan seperti _Data Gathering_, _Data Assessing_, dan _Data Cleaning_. Pada proses Data Gathering, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:

- Duplicate data (data yang serupa dengan data lainnya).
- Missing value (data atau informasi yang "hilang" atau tidak tersedia)
- Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada).

Pada proses _Data Cleaning_ yang dilakukan adalah seperti:
- Converting Column Type (Mengubah tipe suatu kolom).
- Train Test Split (membagi data menjadi data latih dan data uji).
- Normalization (mentransformasi data ke dalam skala yang seragam sehingga semua fitur atau atribut memiliki rentang nilai yang sebanding).

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| NaN | NaN | NaN | NaN |NaN | NaN| NaN	| Created_by_Nidula_Elgiriyewithana  | NaN |

Tabel 2. Melihat data missing value

Pada proyek kasus ini tidak ditemukannya data duplikat, tetapi ditemukannya _missing value_. Adapaun metode yang digunakan untuk mengatasi hal ini adalah dengan menerapkan _Dropping_ yaitu menghapus data yang _missing_ digunakannya metode ini dikarenakan jumlah missing value hanya berjumlah `1`. Lihat _Tabel 2. Melihat data missing value_. Adapun untuk _outlier_ juga dilakukan dengan metode _dropping_ menggunakan metode IQR.  IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumus berikut.

$$IQR = Q_3 - Q_1$$

- Q1 adalah kuartil pertama 
- Q3 adalah kuartil ketiga.

Setelah menggunakan metode IQR untuk menghilangkan _outlier_ pada dataset jumlah dataset menjadi `3790` yang awalnya adalah `4000`.
Pada proyek ini digunakan _Train Test Split_ pada library  *sklearn.model_selection* untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 20:80 dan random state sebesar 60. Pada proyek kasus ini digunakan _Normalization_ pada library _sklearn.preprocessing.MinMaxScaler_ untuk menormalisasi dataset. Semua proses ini diperlukan dalam rangka membuat model yang baik.

## Modeling And Result
Algoritma pada proyek ini melakukan pemodelan dengan 2 algoritma, yaitu:

_Support Vector Machine (SVM)_ adalah algoritma machine learning yang digunakan untuk klasifikasi dan regresi. Algoritma ini bekerja dengan mencari hyperplane yang memisahkan data menjadi dua kelas dengan margin terbesar. Parameter yang digunakan pada SVM kali ini adalah parameter bawaan.
 
 Keuntungan  _Support Vector Machine (SVM)_ :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.
- Dapat digunakan untuk klasifikasi dan regresi.

Kerugian  _Support Vector Machine (SVM)_ :
- Sulit untuk memilih kernel dan parameter lainnya. 
- Sensitif terhadap outlier. 
- Membutuhkan banyak waktu komputasi untuk pelatihan.

_Random Forest_ adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `max_depth` kedalaman maksimum.

Keunggulan _Random Forest_ :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.

Kerugian _Random Forest_ :
- Cenderung overfit pada dataset kecil. 
- Membutuhkan banyak waktu komputasi untuk pelatihan. 
- Sulit untuk diinterpretasikan.

Parameter yang digunakan adalah:
- `kernel` memetakan data input ke ruang dimensi yang lebih tinggi sehingga memungkinkan pemisahan data yang lebih baik.
- `n_estimators` Jumlah pohon keputusan yang akan dibuat dalam ensemble.
- `random_stat` pengambilan sampel secara acak.

## Evaluation
Dalam tahap evaluasi, metrik yang digunakan adalah `accuracy`
Accuracy didapatkan dengan menghitung persentase dari jumlah prediksi yang benar dibagi dengan jumlah seluruh prediksi. Rumus:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

*Penjelasan*
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Rumus ini memecah akurasi menjadi rasio antara data yang diklasifikasikan dengan benar (TP dan TN) dengan jumlah total data. Mengalikan dengan 100% mengubah rasio menjadi persentase.

Berikut hasil accuracy model yang latih:

| Model | Accuracy |
| ------ | ------ |
| SVM | 0.77 |
| RandomForest  | 0.89 |

Tabel 4. Hasil Accuracy

![eval1](https://github.com/user-attachments/assets/fa012554-1ec6-43fd-b6f9-8f878eb66dad)

3a. Model Evaluation SVM

![eval2](https://github.com/user-attachments/assets/ef4dcf05-6331-4bb7-8b0a-101b07d7754b)

3b. Model Evaluation Random Forest

Dilihat dari _Tabel 4. Hasil Accuracy_ dan _Gambar 3a. Model Evaluatin SVM_ dapat diketahui bahwa hasil pemodelan menggunakan algoritma SVM menghasilkan akurasi 77% dengan hasil Confusion Matrix  model yang memiliki total 385 prediksi benar (290 untuk "Bad" dan 295 untuk "Good"), dan 173 prediksi salah (83 False Positive dan 90 False Negative). Juga pada hasil ROC Curve menunjukkan bahwa model memiliki performa yang cukup baik dengan nilai AUC 0.85, yang berarti model memiliki kemampuan yang baik dalam membedakan antara kelas "Good" dan "Bad". Secara keseluruhan, model SVM ini menunjukkan performa yang cukup baik, meskipun masih terdapat beberapa kesalahan dalam prediksi kedua kelas.

Sedangkan, dilihat dari  _Tabel 4. Hasil Accuracy_ dan _Gambar 3b. Model Evaluatin Random Forest_ menghasilkan model dengan akurasi lebih tinggi yaitu 89%. Dari Confusion Matrix, model berhasil membuat 677 prediksi yang benar (341 untuk kelas "Bad" dan 336 untuk kelas "Good"), dan 81 prediksi salah (32 False Positive dan 49 False Negative). Pada ROC Curve menunjukkan performa yang sangat baik dengan nilai AUC sebesar 0.96, yang berarti model ini memiliki kemampuan yang sangat kuat dalam membedakan kelas "Good" dan "Bad". Secara keseluruhan, model Random Forest menunjukkan performa yang sangat baik dengan sedikit kesalahan prediksi, dan kemampuannya untuk memisahkan kelas ditunjukkan oleh AUC yang tinggi. Model ini lebih baik dibandingkan dengan SVM berdasarkan evaluasi ROC dan Confusion Matrix.

Maka dari itu, algoritma Random Forest memiliki Accuracy yang lebih tinggi dengan accuracy 89%. Untuk itu model tersebut yang akan dipilih untuk digunakan. Diharapkan dengan model yang telah dikembangan dapat memprediksi kualitas apel dengan baik menggunakan Random Forest. Alasan mengapa metode Random Forest yang dipilih karena lebih tahan terhadap overfitting, lebih stabil pada data yang kompleks, lebih robust terhadap outliers dan missing data, serta menawarkan interpretasi yang jelas melalui feature importance. Model ini juga cenderung lebih mudah digunakan dan memberikan hasil yang baik tanpa perlu tuning yang ekstensif.

## Referensi
1. Sarnita Sadya.(2022). Produksi Apel Indonesia Sebanyak 509.544 Ton pada 2021.
2. Lomo, Christine P., et al. "Daya Terima Panelist terhadap Kualitas Cider Apel dalam Meningkatkan Nilai Gizi Pangan sebagai Imunitas Tubuh di Pandemi Covid-19." Agrista: Jurnal Ilmiah Mahasiswa Agribisnis UNS, vol. 4, no. 1, 2020, pp. 550-556
3. Wood, T. -.What is a Random Forest?. DeepAI. https://deepai.org/machine-learning-glossary-and-terms/random-forest
4. Gandhi, R. (2018). Support Vector Machine — Introduction to Machine Learning Algorithms: SVM model from scratch. Towards Data Science. https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
