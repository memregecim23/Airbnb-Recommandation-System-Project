import streamlit as st

#projemizin başlık kısmı
st.title(":red[Air]bnb")

#projemizin açıklaması
st.subheader("Kendine uygun konaklama fırsatlarını Keşfet!")

#componentleri sayfa üzerinde düzenleme
left,right,right2 = st.columns(3)

st.write("""Hoş geldiniz! Mükemmel konaklama yerini bulmak için sayısız web sitesinde sonsuza dek arama yapmaktan ve kaybolmaktan yoruldunuz mu? Akıllı Airbnb Öneri Sistemimizle seyahat planlamanıza son dokunuşu yapmak için buradayız. Sistemimiz ihtiyaçlarınızı, tercihlerinizi ve önceliklerinizi anlar ve ardından sizin için en uygun Airbnb ilanlarını sunar. Güvenliğinize, konforunuza ve genel deneyiminize önem veriyoruz, böylece siz de seyahatinizin keyfini çıkarmaya odaklanabilirsiniz.

Nasıl Çalışır? Sistemimiz, geniş bir Airbnb ilanı veri kümesinden yararlanarak ideal konaklama deneyimi yaratmak üzere tasarlanmıştır. Sadece saniyeler içinde, konum, fiyat, olanaklar ve daha fazlası dahil olmak üzere seçtiğiniz kriterlere göre en yüksek puanlı, en beğenilen ve en uygun evleri bulur. Sonsuz kaydırma ve kafa karışıklığına veda edin, mükemmel konaklamanıza merhaba deyin.

Hadi başlayalım! Kriterlerinizi seçin ve girdilerinize dayanarak ideal konaklamanız için en iyi evleri sunalım.

Neden Bizi Seçmelisiniz? Kişiselleştirilmiş Sonuçlar: Size en doğru eşleşmeleri sunmak için standart filtrelerin ötesine geçiyoruz. Hız ve Kolaylık: Saatler süren aramayı sadece birkaç saniyede tamamlayın. Güvenilirlik: Önerilerimiz gerçek kullanıcı yorumlarına ve puanlarına dayanmaktadır; böylece güvenli ve kaliteli bir konaklama yeri bulmanızı sağlarız.""")


#istenilirse bu format ile özelleştirme yapılabir

#açıklamaları sayfanın sol kısmına yazdırma(formatlı bir şekilde)
#left.markdown("")

#görselleri yan yan olarak sayfanın sağ ve sol tarafına yerleştirme
#right.image("", width=350)
right.image("https://www.markafikirleri.com/wp-content/uploads/2020/07/Airbnb-.jpeg", width=500)
#right.image("image.py yada görselin linki", width=250)
left.image("https://gucal.com.tr/images/blog/b-47b6e8a33f.jpg", width=300)


