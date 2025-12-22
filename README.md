![görsel 1](https://github.com/user-attachments/assets/2e4a6b0d-5b47-42a0-97d0-df5b20989cda)
![görsel 2](https://github.com/user-attachments/assets/9aca1423-acc1-4a0d-ab31-d71d914cbee7)
![görsel 3](https://github.com/user-attachments/assets/032e99b7-dc26-4cff-9baa-62f1e340a2da)
![görsel 4](https://github.com/user-attachments/assets/53412041-d896-427b-9087-67ce1c7bea74)
![görsel 5](https://github.com/user-attachments/assets/1acf4ea9-f937-43c5-adb3-70893fb90258)
 
 Not:Aşağıdaki linke tıklayarak canlı olarak demoyu deneyebilirsiniz.Link aynı zamanda about kısmında da bulunmaktadır.
 https://airbnbrecommandationproject.streamlit.app/


# Airbnb Intelligent Recommendation System
Bu proje, kullanıcıların kişisel tercihlerine göre en uygun Airbnb konaklama fırsatlarını bulmalarını sağlayan, veriye dayalı bir web uygulamasıdır. 
İçerisinde veri analitiği paneli, interaktif harita görselleştirmesi ve özelleştirilmiş bir öneri motoru barındırır.

# Proje Amacı ve İş Problemi

Milyonlarca Airbnb ilanı arasında kullanıcıların bütçelerine, konum tercihlerine ve konfor beklentilerine (örneğin; "Manzara", "Jakuzi" vb.) en uygun evi bulması zaman alıcı bir süreçtir.

Bu proje şunları hedefler:
**Veri Analizi:** Airbnb veri setini yükleyerek fiyat, puanlama ve lokasyon bazlı içgörüler sunmak.
**Akıllı Filtreleme:** Kullanıcı dostu arayüz ile ülke, şehir, mahalle ve özellik bazlı dinamik filtreleme sağlamak.
**Görselleştirme:** İlanları interaktif bir harita üzerinde (PyDeck) göstermek ve yoğunluk analizi yapmak.
**Öneri Sistemi:** Kullanıcı kriterlerine göre bir "Custom Score" hesaplayarak en iyi 5 alternatifi sıralamak.

# Özellikler

# Veri Analitiği Modülü (`analyticspage.py`)
**Dinamik Veri Yükleme:** Kullanıcılar kendi `.csv` veya `.zip` dosyalarını yükleyebilir.
**KPI Metrikleri:** Ortalama fiyat, temizlik puanı, doğruluk puanı gibi temel metriklerin anlık hesaplanması.
**Gelişmiş Grafikler:**
    * **Line Chart:** Zaman serisi veya sayısal dağılımlar için (Altair).
    * **Bar Chart:** Kategorik karşılaştırmalar için ölçeklendirilebilir grafikler.

# Harita ve Öneri Modülü (`recommender.py`)
**Coğrafi Filtreleme:** Ülke > Şehir > Mahalle hiyerarşisinde birbirine bağlı filtreler.
**PyDeck Entegrasyonu:** İlanların konumlarını interaktif, zoom yapılabilir bir harita üzerinde kümelenmiş (clustered) veya tekil olarak gösterme.
**Detaylı Filtreler:** Oda tipi, fiyat aralığı, yatak sayısı ve özel olanaklar (Amenities) için slider ve checkbox desteği.
**En İyi 5 Öneri:** Seçilen kriterlere göre hesaplanan skora göre en iyi 5 evi kartlar halinde listeler ve harita üzerinde vurgular ("Focus" özelliği).

# Veri Ön İşleme (`recommandation_system.py`)
Bu proje, ham veriyi işlemek için güçlü bir arka plan algoritması kullanır:
**Kur Dönüşümü:** Farklı ülkelerdeki fiyatları (MXN, THB, TRY vb.) tek bir para birimine endeksleyerek normalize etme.
**Eksik Veri (Missing Value) Yönetimi:**
    * "Studio" dairelerdeki yatak odası sayısını metin madenciliği ile düzeltme.
    * Eksik puanları KNN Imputer veya ortalama ile doldurma.
**Feature Engineering:** İlan açıklamalarından (Amenities) en popüler 20 özelliği (Wi-Fi, Havuz, Klima vb.) çıkarıp Binary değişkenlere dönüştürme.

# Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

**Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/kullaniciadi/airbnb-recommender.git](https://github.com/kullaniciadi/airbnb-recommender.git)
    cd airbnb-recommender
    ```

**Sanal ortam oluşturun (Önerilen):**
    ```bash
    python -m venv .venv
    # Windows için:
    .venv\Scripts\activate
    # Mac/Linux için:
    source .venv/bin/activate
    ```

**Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

**Uygulamayı başlatın:**
    ```bash
    streamlit run main.py
    ```

# Proje Yapısı

```text
├── main.py                     # Uygulama giriş noktası ve navigasyon
├── page/
│   ├── homepage.py             # Karşılama ve tanıtım sayfası
│   ├── analyticspage.py        # Veri analizi ve grafik oluşturma sayfası
│   └── recommender.py          # Harita ve öneri sistemi sayfası
├── recommandation_system.py    # Veri temizleme, EDA ve ML hazırlık scripti
├── Streamlit_App.py            # (Varsa) Custom Streamlit bileşenleri
├── requirements.txt            # Kütüphane bağımlılıkları
└── README.md                   # Proje dokümantasyonu

** Not: Proje yapısı page sayfası oluşturulmadan da kullanılabilir gerekli dosyaların aynı klasöre olması yeterlidir.Aynı şeklide streamlit ile ayağa kaldırmak için main.py
çalıştırılır. **

 **Geliştiriciler : Benimde içinde bulunduğum bir ekiple beraber yazılmıştır.**
