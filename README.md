# Turkish Markov

Büyük Türkçe metin derlemleri (korpusları) için akışlı (streaming) ve disk tabanlı bir Markov zinciri aracı.
Derlemi veya geçiş grafiğini RAM'e yüklemeden kelime düzeyinde cümle modelleri ve karakter düzeyinde sahte kelime (pseudo-word) modelleri oluşturur.

Projenin Python 3.10+ ve SQLite (Python ile birlikte gelir) dışında hiçbir çalışma zamanı (runtime) bağımlılığı yoktur. Tüm çoklu işlem (multiprocessing) giriş noktaları modül düzeyinde olduğu ve yürütülebilir giriş noktası korunduğu için Windows üzerinde güvenle çalıştırılabilir.

## Veri Seti ve Önceden Eğitilmiş Veritabanı / Dataset & Pre-trained Database

GitHub'ın dosya boyutu sınırları nedeniyle, büyük veri seti ve önceden indekslenmiş veritabanı **Hugging Face** üzerinde barındırılmaktadır:
> 🔗 **Hugging Face Dataset:** [https://huggingface.co/datasets/beert00/turkish-markov](https://huggingface.co/datasets/beert00/turkish-markov)

Projeyi kullanmak için aşağıdaki iki seçenekten birini tercih edebilirsiniz:

- **Seçenek A - Hızlı Başlangıç (Option A: Quick Start):**
  - Doğrudan **[`turkish.db` (Direct Link)](https://huggingface.co/datasets/beert00/turkish-markov/resolve/main/turkish.db)** dosyasını indirin.
  - Dosyayı proje ana dizinine (root directory) yerleştirin (gerekiyorsa adını `turkish` olarak değiştirin / *renaming to `turkish` if required*) ve hemen çalıştırmaya başlayın.

- **Seçenek B - Sıfırdan Eğitme/İndeksleme (Option B: Train/Build from Scratch):**
  - Veri setini incelemek veya sıfırdan model eğitmek/indekslendirmek isterseniz **[`birlesmis.txt` (Direct Link)](https://huggingface.co/datasets/beert00/turkish-markov/resolve/main/birlesmis.txt)** dosyasını indirin.
  - İndirdiğiniz ham derlem ile CLI üzerinden yeniden indeksleme yapabilirsiniz.

## Windows grafik arayüzü

Uygulamayı bir terminal penceresi olmadan tarayıcınızda açmak için `arayuz.pyw` dosyasına çift tıklayın. Yalnızca yerel bilgisayarınızda çalışır ve derlemi veya üretilen metni hiçbir yere yüklemez. İlk sekme `turkish.db` veritabanından metin üretir; ikinci sekme yeni bir veritabanı oluşturur ve ilerlemeyi görüntüler.

## Hızlı başlangıç

Kaynak dizininden:

```powershell
python main.py build-db --input-dir . --order 2 --db turkish.db
python main.py generate --db turkish.db --prompt "İstanbul" --length 50 --temperature 0.8
```

Proje gereksinimlerinde istenen tam minimal komut da çalışır ve `markov.db` dosyasını yazar:

```powershell
python main.py build-db --input-dir C:\corpora\turkish --order 2
python main.py generate --prompt "başlangıç" --length 50 --temperature 1.0
```

Bir `turkish-markov` konsol komutu yüklemek için:

```powershell
python -m pip install .
turkish-markov --help
```

## Modeller

`--order N`, bir durumun (state) önceki **N birimi** içerdiği ve her satırın bir sonraki birimin frekansını sakladığı anlamına gelir. Bu nedenle derece (order) 2, iki belirteçli (token) bir durumu ve onun bir sonraki belirtecini (bir trigram geçişi) saklar. Varsayılan değer derece 2'dir; derece 3 genellikle yerel akıcılığı artırır ancak kelime veritabanını çok daha büyük hale getirebilir.

Varsayılan derleme modu `word` (kelime) modudur. Gerektiğinde modelleri açıkça seçin:

```powershell
# Her iki model tek bir veritabanında
python main.py build-db --input-dir C:\corpora --db turkish.db --mode both --order 2

# Yalnızca karakter modeli
python main.py build-db --input-dir C:\corpora --db words.db --mode char --order 3

# "ışı" ile başlayan bir sahte kelime (pseudo-word) üretin
python main.py generate --db words.db --mode char --prompt "IŞI" --length 12 --temperature 0.9
```

Kelime modunda `--length`, yeni üretilen sözlü belirteçlerin (lexical tokens) sayısıdır; noktalama işaretleri kotayı tüketmez. Karakter modunda ise önekten (prefix) sonraki maksimum yeni karakter sayısıdır ve üretim, öğrenilen bir kelime sınırında daha erken durabilir. `--temperature 0` en sık gerçekleşen geçişi seçer; `1.0` gözlemlenen dağılımdan örnekleme yapar; daha yüksek değerler dağılımı düzleştirir. `--seed` rastgele örneklemeyi tekrarlanabilir kılar.

## Büyük derlem (large-corpus) mimarisi

Derleme hattı (build pipeline), 10 GB'lık tek bir kitap dökümü gibi büyük derlemler için tasarlanmıştır:

1. Girdi keşfi, özyinelemeli (recursively) olarak `.txt` dosyalarını seçer. Tek bir dosya yolu da kabul edilir.
2. Büyük dosyalar, istenen yığın (chunk) boyutuna yakın bayt aralıklarına bölünür. Aralıklar boş satır paragraf sınırlarına göre hizalanır, böylece kaydırılmış satırlar aynı paragrafın parçası olarak kalır. Sınırlı bir geri çekilme (fallback) mekanizması, paragraf sınırı olmayan bozuk dosyaları işler.
3. Başlatılan alt süreçler (spawned processes), UTF-8'i kademeli (incrementally) olarak çözer, Türkçe metni normalize eder ve sınırlı sayıdaki benzersiz geçişleri toplar.
4. Her işçi (worker) süreci kendine özel geçici bir SQLite parçasına (shard) sahiptir. Bu, birden fazla süreç tek bir SQLite veritabanına UPSERT işlemi yaptığında oluşan tek yazıcı (single-writer) çakışmasını önler.
5. Ana süreç (parent), bu parçaları bir hazırlık (staging) veritabanında atomik olarak birleştirir, sıkıştırılmış bir somutlaştırılmış (materialized) geri çekilme (backoff) tablosu oluşturur, WAL denetim noktası (checkpoint) koyar, indeksleme sırasında hiçbir kaynağın değişmediğini doğrular ve son olarak istenen çıktı yolunu değiştirir.

RAM kullanımı esas olarak `workers × (batch-size + largest paragraph)` ile sınırlıdır. Yararlı ince ayar (tuning) seçenekleri şunlardır:

```text
--workers N             0 otomatik olarak 8'e kadar seçer; CPU ve depolama için ayarlayın
--chunk-size-mb N       hedef dosya aralığı, varsayılan 256 MiB
--batch-size N          işçi başına benzersiz geçiş sayısı, varsayılan 100.000
--max-paragraph-mb N    bozuk kesintisiz kayıtlar için güvenlik sınırı, varsayılan 16
--temp-dir PATH         işçi SQLite parçalarının konumu
```

Geçici parçalar ve nihai veritabanı birleştirme sırasında bir arada bulunur. Yeterli boş disk alanı bulundurun: derece 3 kelime modeli orijinal derlem boyutunu aşabilir ve `both` (her iki) modda derleme yapmak daha fazla geçici alan gerektirir. SSD/NVMe depolama, parça birleştirmeyi önemli ölçüde hızlandırır. Mevcut veritabanları hiçbir zaman sessizce karıştırılmaz; atomik olarak değiştirmek için `--overwrite` parametresini geçin.

Yayınlanan veritabanı WAL modunu, `synchronous=NORMAL`, hazırlanmış toplu (batched) UPSERT'leri, `WITHOUT ROWID` geçiş/backoff tablolarını ve okuma tarafı mmap/önbelleğini kullanır. Olasılıklar kayan noktalı (floating-point) sayılar yerine frekans sayıları olarak saklanır; üretim, tam ağırlıklı dağılımı türetir ve sıcaklığı log alanında uygulayarak alt taşmayı (underflow) önler ve sıcaklığın istek başına değişmesine olanak tanır.

## Türkçe normalizasyon

Tüm dosyalar UTF-8 olarak çözülür. Varsayılan `strict` politikası, bozuk girdilerde bir yol ve bayt aralığı hatası vererek durur; bilerek kayıplı derlemler için `--encoding-errors replace` seçeneği mevcuttur ve değiştirme sayısını raporlar.

Normalizasyon işlemi Unicode NFKC/NFC, kanonik kesme işaretleri/tırnaklar/tireler, üç nokta normalizasyonu, kontrol karakterlerinin kaldırılması ve boşlukların birleştirilmesini (collapse) uygular. Türkçe harf dönüştürme, sıradan Unicode küçük harfe çevirme işleminden önce açıkça `I → ı` ve `İ → i` eşlemesini yapar. Düzeltme işaretli Türkçe harfler (`â`, `î`, `û`) korunur. Aynı hat istemler (prompt) için de uygulanır.

Kelime (word) modu, kelime içi kesme işaretlerini ve tireleri, ayrıca yararlı cümle noktalamalarını korur. Cümle, paragraf ve dosya sınırlarında sıfırlanır. Karakter (char) modu, her katı Türkçe kelimeyi başlangıç/bitiş gözlemcileriyle (sentinels) ayrı ayrı öğrenir; kesme işaretli ekleri birleştirir, tireli birleşik kelimeleri ayırır ve asla bir kelimeden diğerine geçiş oluşturmaz. 64 harften uzun belirteçler (token), bozuk OCR/kesintisiz kayıt olarak kabul edilir ve karakter modeli tarafından atlanır.

## Proje yapısı

```text
turkish_markov/
  cleaner.py    Türkçe normalizasyon, belirteçleştirme (tokenization), belirteç birleştirme (detokenization)
  database.py   SQLite şemaları, parça birleştirme, geçiş alma
  indexer.py    akış/yığınlama ve çoklu işlem derleme hattı
  markov.py     backoff, sıcaklık ölçeklendirme, ağırlıklı örnekleme
  cli.py        build-db ve generate komutları
main.py         korumalı kaynak ağacı giriş noktası
tests/          cleaner, database, generator ve CLI testleri
```

Bağımlılıksız test paketini çalıştırmak için:

```powershell
python -m unittest discover -v
```

Normal testler küçük geçici derlemler kullanır. Çok gigabaytlık derlem test paketi tarafından hiçbir zaman okunmaz.

