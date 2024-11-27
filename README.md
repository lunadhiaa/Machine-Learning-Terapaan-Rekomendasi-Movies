# Laporan Proyek Machine Learning - Lulu Nadhiatun Anisa

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
- Bagaimana cara membuat sistem rekomendasi film yang memungkinkan film yang disukai oleh satu pengguna dapat direkomendasikan kepada pengguna lain dengan preferensi serupa?
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

### Variabel-variabel pada dataset:
Pada file `movies.csv` berisi daftar film yang memiliki 9742 data dan 3 feature:

- `movieId` : memuat nomor ID film  
- `title` : memuat judul film
- `genres` : memuat genre film

Pada file `ratings.csv` berisi daftar ratings atau penilaian terhadap satu film yang memiliki 100836 data dan 4 feature:

- `userId` : memuat nomor ID users
- `movieId` : memuat nomor ID film  
- `rating` : memuat rating atau penilaian films dalam skala bintang, dengan peningkatan setengah bintang dalam rentang 0,5 - 5 bintang
- `timestamp` : memuat kode timestamp

Pada file `tags.csv` berisi daftar tags pada film yang diberikan users yang memiliki 3683 data dan 4 feature:

- `userId` : memuat nomor ID users
- `movieId` : memuat nomor ID film  
- `tag` : memuat tag film
- `timestamp` : memuat kode timestamp

Pada file `links.csv` berisi daftar links film yang mengarah ke laman website films yang memiliki 9742 dan 3 feature: 

- `movieId` : memuat nomor ID film yang merujuk pada website MovieLens
- `imdbId` : memuat nomor ID film yang merujuk pada website IMDb
- `tmdbId` : memuat nomor ID film yang merujuk pada website TMDB

### Exploratory Data Analysis (EDA)
Proses eksplorasi data (Exploratory Data Analysis/EDA) dilakukan dengan menganalisis dataset secara mendalam untuk memperoleh pemahaman yang komprehensif mengenai karakteristik data, sehingga dapat mengungkapkan insight dan pengetahuan (knowledge).
**Univariate Analysis**
Merging Data adalah proses menggabungkan dua atau lebih set data yang memiliki atribut atau kolom yang relevan, untuk membentuk satu dataset yang konsisten dan utuh. Tujuan dari merging data adalah untuk mengintegrasikan informasi yang tersebar dalam berbagai sumber atau tabel sehingga dapat digunakan dalam analisis atau pemodelan. Berikut ini beberapa tahapan yang dilakukan untuk dalam tahapan merging data yang saya lakukan yaitu:
- Menggabungkan beberapa file dengan fungsi `np.concatenate` berdasarkan pada movieId dengan menyimpanya pada variabel `movie_all`
![1](https://github.com/user-attachments/assets/5b19ee23-45e0-4ef6-970f-9d672432c3f7)

- Menggabungkan beberapa file dengan fungsi `np.concatenate` berdasarkan pada userId dengan menyimpanya pada variabel `user_all`
![2](https://github.com/user-attachments/assets/01988124-1995-4ff4-be25-e95d4f80ded9)

- Menggabungkan beberapa file seperti `links`, `movies`, `ratings`, dan `tags` dengan menyimpannya pada variabel `movie_info`
![3](https://github.com/user-attachments/assets/fd615154-ca95-49d3-bdba-62751fc9c079)

- Menggabungkan dataframe ratings dengan `movie_info` Berdasarkan Nilai `movieId`
![4](https://github.com/user-attachments/assets/4ef808ef-f031-41e7-b28b-ff0a6a2e05c2)

- Menggabungkan data dengan featuers `movies`
![5](https://github.com/user-attachments/assets/7475e398-ed72-40e8-8bcf-402662cce922)

## Data Preparation


Setelah selesai melakukan Merging data, dapat melanjutkannya dengan melakukan beberapa tahapan sebagai berikut yaitu : 
- Mengatasi missing value : menyeleksi data apakah data tersebut ada yang kosong atau tidak, jika ada data kosong maka saya akan.menghapusnya
- Menggabungkan variabel : untuk menggabungkan beberapa variabel berdasarkan id yang sifatnya unik (berbeda dari yang lain).
- Mengurutan data : untuk mengurutkan data berdasarkan movieId secara asceding.
- Mengatasi duplikasi data : untuk mengatasi data yang nilai atau isinya sama.
- Konversi data menjadi list : untuk mengubah data menjadi list
- Membuat dictionary : Untuk membuat dictionary dari data yang ada.
- Mapping data : untuk memetakan data
  
## Modeling dan Result
## Evaluation
## Referensi
1. antaranews.com (2020, 10 Oktober). Layanan "streaming", tantangan dan peluang perfilman Indonesia. Diakses pada 25 November 2024, dari https://www.antaranews.com/berita/1776245/layanan-streaming-tantangan-dan-peluang-perfilman-indonesia#google_vignette
2. kumparan.com. (2021, 31 Agustus). Mengenal Istilah Binge Watching dan Hopping yang Jadi Tren Terbaru Nonton Film. Diakses pada 25 November 2024, dari https://kumparan.com/millennial/mengenal-istilah-binge-watching-dan-hopping-yang-jadi-tren-terbaru-nonton-film-1wREvjSoXeB
3. Arfisko, Hilmi Hidayat dan Wibowo, Agung Toto. 2022. “Sistem Rekomendasi Film Menggunakan Metode Hybrid Collaborative Filtering Dan Content-based Filtering”. *e-Proceeding of Engineering*, Vol.9, No. 3 Juni 2022: 2159.
4. Wiputra, Michael M., and Yusup J. Shandi. "Perancangan Sistem Rekomendasi Menggunakan Metode Collaborative Filtering dengan Studi Kasus Perancangan Website Rekomendasi Film". *Media Informatika*, vol. 20, no. 1, 2021, pp. 1-18, doi:[10.37595/mediainfo.v20i1.54](https://dx.doi.org/10.37595/mediainfo.v20i1.54).

