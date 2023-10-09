### Customer Segmentation with RFM ###

## Business Problem

# FLO musterilerini segmentlere ayirip bu segmentlere gore pazarlama stratejileri belirlenmek isteniyor.

## Dataset Info

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


## Data Understanding

import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)

df_ = pd.read_csv("/Users/betulyilmaz/Documents/Miuul/CRM Analytics/Case Study 1-FLO Musteri Segmentasyonu/flo_data_20k.csv")
df = df_.copy() # Datasetin bir kopyasini aliyoruz, okumasi vakit aldigindan ve sonrasinda olasi bir durum icin yedekte tutmak istiyoruz.
df.head()

df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

# Herbir musterinin toplam alisveris sayisini hesapliyoruz.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Degisken tiplerini inceliyoruz, tarih olan degiskenlerin tipini date'e cevirmemiz gerekiyor.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# Alisveris kanallarina gore, musteri sayisi, toplam yapilan alisveris sayisi ve toplam harcamalarini inceliyoruz.
df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})

# En fazla kazanc getiren ilk 10 musteriyi siraliyoruz.
df.sort_values("customer_value_total", ascending=False)[:10]

# En fazla alisveris yapan ilk 10 musteriyi siraliyoruz.
df.sort_values("order_num_total", ascending=False)[:10]

# Veri on hazirlama surecini fonksiyonlastiriyoruz.
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df


## Calculating RF and RFM Scores

# Veri setindeki en son alisveris yapilan tarihten 2 gun sonrasini analiz tarihi olarak belirliyoruz.
df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

# customer_id, recency, frequnecy ve monetary degerlerini ve yer aldigi yeni bir RFM dataframe olusturuyoruz
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

# Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasinda skorlara cevrilmesi ve
# bu skorlarin recency_score, frequency_score ve monetary_score olarak kaydedilmesi
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# recency_score ve frequency_score’u tek bir degisken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# 3. recency_score ve frequency_score ve monetary_score'u tek bir degisken olarak ifade edilmesi ve RFM_SCORE olarak kaydedilmesi
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))


## RF segmentation

# Olusturulan RFM skorlarin daha aciklanabilir olmasi icin segment tanimlama ve tanimlanan seg_map yardimi ile RF_SCORE'u segmentlere cevirme
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)


## Action!

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydedelim.

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

# Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydedelim.

target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)

