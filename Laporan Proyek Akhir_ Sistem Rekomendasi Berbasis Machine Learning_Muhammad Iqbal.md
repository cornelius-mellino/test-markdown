# Laporan Proyek Akhir: Sistem Rekomendasi Berbasis Machine Learning

## Project Overview

Sistem rekomendasi ini dirancang untuk membantu pengguna menemukan item yang relevan berdasarkan deskripsi konten. Proyek ini menggunakan pendekatan *content-based filtering* dengan pemrosesan data menggunakan *TF-IDF vectorization* dan metrik kesamaan kosinus untuk menghitung tingkat kesamaan antar item.

## Business Understanding

Sistem rekomendasi sangat penting dalam membantu pengguna menyaring informasi dalam jumlah besar. Contohnya, dalam domain seperti:

- Rekomendasi film untuk penonton.
- Rekomendasi produk untuk e-commerce.
- Rekomendasi buku bagi pembaca.

Proyek ini bertujuan untuk:

1. Membantu pengguna menemukan item yang serupa dengan minat mereka.
2. Mengoptimalkan pengalaman pengguna melalui personalisasi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "TMDB 5000 Movie Dataset" dari Kaggle. Berikut adalah beberapa informasi penting:

- **Struktur Dataset**: Dataset memiliki kolom `title` (judul film) dan `overview` (deskripsi film).
- **Ukuran Dataset**: Dataset berisi 4803 film dengan informasi deskripsi yang digunakan untuk ekstraksi fitur.
- **Kebersihan Data**: Proses *data cleaning* mencakup:
  - Menghapus data duplikat.
  - Mengatasi nilai yang hilang dengan menghapus entri kosong pada kolom `overview`.

Contoh dataset:

```
| Title               | Overview                           |
|---------------------|------------------------------------|
| The Dark Knight     | Batman raises the stakes...       |
| Inception           | A thief who steals corporate...  |
```

## Data Preparation

- **Ekstraksi Fitur**: Menggunakan *TF-IDF Vectorizer* untuk mengubah deskripsi menjadi representasi vektor numerik.
- **Normalisasi Data**: Proses ini memastikan bahwa setiap deskripsi diberi bobot sesuai relevansinya.
- **Penghitungan Kesamaan**: Menggunakan metrik kesamaan kosinus untuk menghitung tingkat kemiripan antara deskripsi film.

Kode utama:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(dataset['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

## Modeling

Model ini menggunakan pendekatan *content-based filtering* yang bekerja dengan langkah berikut:

1. **Input**: Judul film yang ingin direkomendasikan.
2. **Proses**: Model mencari film serupa berdasarkan skor kesamaan kosinus terhadap deskripsi.
3. **Output**: Daftar rekomendasi film yang relevan.

Contoh fungsi rekomendasi:

```python
def recommend(title, cosine_sim=cosine_sim):
    idx = dataset[dataset['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return dataset['title'].iloc[movie_indices]
```

Hasil rekomendasi untuk input "Avatar":

1.  Apollo 18
2.  The Matrix
3.  Hanna
4.  Semi-Pro
5.  Aliens vs Predator: Requiem
6.  E.T. the Extra-Terrestrial
7.  Beowulf
8.  Dolphin Tale 2
9.  Transformers: Age of Extinction
10. American Gangster

## Evaluation

Model dievaluasi menggunakan pendekatan *manual verification*, yaitu:

1. Memeriksa kesesuaian rekomendasi dengan film yang dimasukkan.
2. Melihat tingkat relevansi berdasarkan kemiripan deskripsi.

Kesimpulan: Model memberikan hasil yang relevan berdasarkan deskripsi film yang dimasukkan.

## Summary

Proyek ini berhasil membangun sistem rekomendasi *content-based filtering* yang dapat diterapkan untuk berbagai kasus seperti film, buku, dan produk. Sistem ini menggunakan representasi vektor dari deskripsi item untuk menemukan item serupa.

### File Submission

- File Jupyter Notebook (.ipynb) yang berisi kode dan hasil eksekusi.
- File Python (.py) untuk implementasi model.
- Laporan ini dalam format Markdown (.md).

**Saran Pengembangan**:

- Mengintegrasikan sistem ini dengan UI/UX untuk pengguna akhir.
- Menggabungkan dengan pendekatan *collaborative filtering* untuk hasil yang lebih baik.

**Referensi Dataset**:

- Kaggle: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

