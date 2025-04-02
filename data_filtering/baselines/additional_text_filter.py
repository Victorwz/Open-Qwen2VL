import re

fasttext_levels = [
        { # keep 73.46% data, on one parquet file
         'en': -1, # in case fasttext may return score as -0.003
         '8lang': -1,
         'others': 0.2,
        }, 
        { # keep 71.54% data, on one parquet file
         'en': -1,
         '8lang': -1,
         'others': -1,
        },
        { # keep 69.99% data, on one parquet file
         'en': -1,
         '8lang': 0.2,
         'others': -1,
        },
        { # keep 64.69% data, on one parquet file
         'en': -1,
         '8lang': 0.4,
         'others': -1,
        },
        { # keep 62.52% data, on one parquet file
         'en': 0.2,
         '8lang': 0.4,
         'others': -1,
        },
        { # keep 52.12% data, on one parquet file
         'en': 0.4,
         '8lang': 0.4,
         'others': -1,
        },
        { # keep 47.08% data, on one parquet file
         'en': 0.2,
         '8lang': 2, # in case fasttext may return score as 1.0002
         'others': -1,
        },
        { # keep 36.68% data, on one parquet file
         'en': 0.4,
         '8lang': 2,
         'others': -1,
        },
        ]

nine_lang = set([
    'de',
    'fr',
    'es',
    'it',
    'pt',
    'nl',
    'vi',
    'tr',
    'en'
    ])

medium_high_freq_bad_text = {
"image": 545496,
"nan": 224766,
"identicon": 190196,
"untitled image": 153567,
"avatar": 142656,
"front cover": 129085,
"captcha": 107723,
"product image": 102887,
"photo": 85011,
"thumbnail": 80608,
"*": 72113,
"product details": 70032,
"[​img]": 61335,
"画像": 59407,
"img": 58386,
"nan": 58365,
"1": 53725,
"main product photo": 52035,
"image 1": 51689,
"superthumb": 51522,
"gallery image": 48094,
"photo:": 45680,
"no title": 43437,
"additional product image": 39508,
"logo": 36617,
"foto": 35556,
"2": 33398,
"изображение": 31743,
"gallery photo": 31456,
"portada": 31147,
"cover": 30055,
"screenshot image": 27165,
"3": 26373,
"antispam": 26194,
"...": 26132,
"user avatar": 25271,
"print image": 25054,
"code": 25041,
"map": 23368,
"untitled photo": 22906,
"#########": 22450,
"détails sur le produit": 22060,
"captcha image": 21973,
"product": 21945,
"undefined": 21885,
"4": 21694,
"#": 20867,
"thumb": 20741,
"qr code": 20710,
"profile photo": 20430,
"thumbnail image": 20391,
"image description": 20191,
"imagen": 20011,
"bild": 19764,
"photo of this room": 19342,
"5": 18052,
"写真": 17501,
"featured image": 17321,
"produkt-information": 17209,
"图片": 16826,
"page 1": 16509,
"icon": 15793,
".": 15551,
"[img]": 15324,
"[ocr errors]": 15172,
"listing": 15053,
"none": 14944,
"6": 14884,
"alt": 14797,
"товар": 14541,
"image preview": 14487,
"shopfoodex, co. bbb business review": 14456,
"listing image": 14398,
"foto immobile": 14256,
"-": 14029,
"image.png": 13712,
"gallery": 13666,
"7": 13002,
"untitled": 12986,
"wpdiscuz_captcha": 12548,
"picture": 12255,
"0": 11853,
"listing image 1": 11418,
"slider": 11320,
"8": 11252,
"смотреть онлайн видео": 11060,
"aeproduct.getsubject()": 11005,
"12": 10846,
"10": 10740,
"post image": 10610,
"11": 10349,
"listing image 2": 10345,
"images": 10209,
"cover art": 10204,
"sale picture": 10195,
"author": 10146,
"listing image 3": 10022,
"kitchen": 10013,
"a": 9761,
"qrコード": 9749,
"9": 9676,
"listing image 4": 9652,
"listing image 5": 9575,
"1.jpg": 9513,
"imagem": 9459,
"blavatar": 9121,
"listing image 6": 9101,
"listing image 7": 9084,
"item image": 8999,
"gravatar": 8766,
"商品画像": 8676,
"listing image 8": 8635,
"image feature": 8580,
"sharecg home": 8572,
"screenshot": 8461,
"商品の詳細": 8190,
"qrcode": 8159,
"roll over large image to magnify, click large image to zoom": 8148,
"picture 1 of 1": 8132,
"avatar image": 8097,
"picture 2": 8046,
"pic": 8015,
"poster": 7998,
"review image": 7967,
"listing image 9": 7958,
"image 2": 7949,
"物件画像": 7848,
"listing image 10": 7730,
"image and video hosting by tinypic": 7723,
"preview": 7678,
"photo 1": 7616,
"video thumbnail": 7604,
"listing image 11": 7530,
"13": 7470,
"listing image 12": 7426,
"video": 7421,
"placeholder": 7347,
"listing image 13": 7214,
"voorkant": 7175,
"thumbnail for": 7152,
"enter image description here": 7134,
"外観": 7120,
"[graphic]": 7070,
"2.jpg": 7047,
"download": 7019,
"listing image 14": 6958,
"[verification code]": 6855,
"\\": 6852,
"afbeelding": 6837,
"listing image 15": 6806,
"www.net.kg": 6749,
"immagine": 6734,
"14": 6728,
"front": 6668,
"user": 6636,
"qr": 6614,
"home": 6614,
"next page": 6612,
"figure 1": 6560,
"customer image": 6540,
"/": 6534,
"listing image 16": 6520,
"zvětšit fotografii": 6511,
"file": 6490,
"photobucket": 6397,
"download section": 6397,
"thumbnail photo": 6358,
"cover image": 6294,
"头像": 6214,
"фото": 6170,
"posted image": 6170,
"listing image 17": 6168,
"1": 6165,
"listing image 18": 6152,
"картинка с кодом": 6136,
"digitized item thumbnail": 6118,
"[blocks in formation]": 6104,
"card image cap": 6081,
"picture 3": 6041,
"photo 2": 6021,
"15": 5943,
"3.jpg": 5916,
"picture 1": 5909,
"facebook": 5887,
"profile cover photo": 5856,
"profile picture": 5738,
"listing image 19": 5731,
"obrázek": 5729,
"title": 5713,
"picture 4": 5695,
"listing image 20": 5679,
"click here": 5669,
"tumblr media": 5666,
"photo 3": 5626,
"figure 2": 5624,
"article image": 5609,
"twitter": 5559,
"photo0.jpg": 5488,
"listing image 21": 5444,
"16": 5423,
"在这里插入图片描述": 5402,
"vector image": 5388,
"image.jpg": 5378,
"image for post": 5338,
"image01": 5338,
"間取図": 5315,
"profile image": 5312,
"picture 5": 5287,
"الغلاف الأمامي": 5283,
"banner": 5250,
"listing image 22": 5164,
"picture 6": 5162,
"photo 4": 5114,
"image 3": 5064,
"4.jpg": 5053,
"the article as it originally appeared.": 5053,
"listing image 23": 5012,
"index": 4972,
"code anti-spam": 4936,
"scan this!": 4934,
"hidden seo image": 4919,
"photo 5": 4858,
"the captcha image": 4838,
"listing image 24": 4819,
"picture 7": 4726,
"wall view 002": 4720,
"17": 4708,
"23": 4684,
"wall view 001": 4672,
"fulltext thumbnail": 4670,
"no image": 4653,
"figure 3": 4590,
"view media": 4558,
"click here for mcadcafe": 4537,
"items.[0].image.alt": 4501,
"170x170": 4494,
"stock photo": 4489,
"18": 4472,
"listing image 25": 4461,
"zoom": 4429,
"表紙画像": 4406,
"사용자 삽입 이미지": 4363,
"clients": 4328,
"photo 6": 4309,
"[商品価格に関しましては、リンクが作成された時点と現時点で情報が変更されている場合がございます。]": 4308,
"product name": 4290,
"copertina anteriore": 4254,
"first slide": 4249,
"イメージ": 4246,
"2": 4239,
"wall view 003": 4215,
"listing image 26": 4181,
"picture 8": 4173,
"profile": 4170,
"click here to see the photo!": 4163,
"my photo": 4102,
"5.jpg": 4067,
"код": 4065,
"original": 4049,
"listing image 27": 4042,
"main image": 4026,
"19": 4016,
"exterior": 4001,
"image 0": 3999,
"封面": 3998,
"20": 3947,
"photo 7": 3931,
"foto imagen": 3912,
"picture 9": 3904,
"photo 8": 3886,
"post": 3879,
"item-thumbnail": 3864,
"listing image 28": 3830,
"gravatar image": 3824,
"pdf extract preview": 3791,
"slide": 3778,
"image1": 3769,
"chania": 3765,
"imagebam.com": 3765,
"flyin.com": 3741,
"product view": 3725,
"logotipas": 3700,
"photo 9": 3697,
"placeholder image": 3693,
"detalles del producto": 3667,
"22": 3645,
"image 4": 3625,
"default": 3613,
"instagram": 3612,
"imagethumbnail": 3610,
"fullsizerender": 3606,
"product photo": 3594,
"figure 4": 3593,
"listing image 29": 3568,
"3": 3552,
"front image": 3540,
"responsive image": 3521,
"click to enlarge": 3506,
"аватар": 3506,
"6.jpg": 3500,
"21": 3497,
"dettagli prodotto": 3482,
"ebay producr": 3415,
"photo 10": 3406,
"product-image": 3405,
"picture 10": 3395,
"post-thumb": 3390,
"no description": 3364,
"物件写真": 3348,
"キャプチャ画像": 3343,
"photo1.jpg": 3324,
"защитный код": 3298,
"product thumbnail": 3297,
"мастер рекламы": 3297,
"imagen de galería": 3224,
"listing image 30": 3209,
"produktbild": 3204,
"cover photo": 3177,
"big image": 3171,
"image thumbnails": 3158,
"bild 1 von 1": 3149,
"computersalg.dk": 3140,
"código qr": 3114,
"передняя обложка": 3105,
"その他": 3103,
"4": 3072,
"album cover": 3054,
"オススメ物件": 3046,
"small": 3041,
"間取り図": 3027,
"main": 3020,
"listing image 31": 3007,
"table 1": 2987,
"image host": 2987,
"your current avatar": 2972,
"24": 2961,
"gallery image of this property": 2958,
"copertina": 2935,
"preview image 1": 2924,
"close": 2923,
"7.jpg": 2919,
"foto de foto": 2912,
"listing image 32": 2908,
"gambar gravatar": 2890,
"zalando": 2877,
"$title$": 2874,
"1": 2857,
"1.png": 2852,
"picture 11": 2846,
"앞표지": 2843,
"상품 이미지1": 2831,
"纯奶手撕吐司的做法 步骤1": 2818,
"view larger image": 2805,
"captcha-bild": 2789,
"item": 2787,
"item preview": 2779,
"5": 2772,
"image captcha": 2770,
"dav": 2769,
"background": 2766,
"product image 1": 2757,
"przednia okładka": 2756,
"maincover": 2754,
"página siguiente": 2738,
"thumbnail 1": 2735,
"这间客房的照片": 2728,
"thumb original": 2727,
"image 5": 2723,
"image-empty-state.png": 2711,
"play preview video": 2702,
"name": 2696,
"első borító": 2689,
"二维码": 2681,
"11.jpg": 2674,
"company logo": 2667,
"vector image vector image": 2666,
"微信扫一扫": 2651,
"imagen no disponible": 2648,
"unnamed": 2639,
"상품이미지": 2629,
"i": 2625,
"listing image 33": 2623,
"listing photo": 2621,
"photo detail": 2615,
"image title": 2603,
"d": 2601,
"8.jpg": 2596,
"test": 2580,
"photo thumbnail": 2580,
"photo de l'hébergement": 2572,
"listing image thumbnail": 2563,
"25": 2562,
"listing image 34": 2555,
"profile picture?fit=crop&height=50&width=50": 2544,
"medium": 2540,
"предња корица": 2532,
"user posted image": 2532,
"aa": 2530,
"alt text": 2526,
"big picture": 2519,
"untitled prezi": 2518,
"picture 12": 2512,
"10.jpg": 2508,
"schwarz": 2504,
"view details": 2504,
"item images": 2502,
"���ʲ���": 2499,
"{singlekeyword} {attribute}": 2473,
"6": 2472,
"figure 5": 2469,
"image loading..": 2463,
"?? lbl.alttext.altthumbnailimage ??": 2456,
"back": 2453,
"captcha изображение": 2432,
"il_170x135": 2429,
"keyword": 2405,
"immagine captcha": 2399,
"hyperleap image": 2392,
"26": 2391,
"full size": 2386,
"s": 2385,
"alternate text": 2380,
"photo2.jpg": 2376,
"код проверки": 2373,
"google": 2369,
"foto von diesem zimmer": 2368,
"large": 2364,
"アバター": 2355,
"immagine prodotto": 2350,
"listing image 35": 2347,
"etukansi": 2335,
"admin": 2334,
"(null)": 2334,
"間取り": 2329,
"2": 2329,
"..": 2327,
"foto 1": 2315,
"rss": 2303,
"m": 2298,
"72677": 2290,
"фотография этого номера": 2289,
"2.png": 2284,
"cision": 2259,
"商品写真": 2256,
"7": 2248,
"כריכה קדמית": 2240,
"kohdekuva": 2238,
"imagen captcha": 2232,
"9.jpg": 2232,
"comment-avatar": 2224,
"[image]": 2224,
"27": 2223,
"foto de esta habitación": 2222,
"listing image 36": 2218,
"click for larger image": 2207,
"inventory item": 2188,
"see this image larger": 2187,
"to do: alt text": 2182,
"image of item": 2175,
"slide1": 2173,
"detail": 2166,
"item.title": 2160,
"album photo": 2151,
"включите картинки!": 2149,
"user image": 2141,
"listing item": 2140,
"12.jpg": 2140,
"предна корица": 2135,
"single-product": 2129,
"房源图": 2119,
"30": 2118,
"фотография": 2106,
"photo-maison": 2105,
"image 6": 2103,
"d�tails sur le produit": 2102,
"boligpresentationsbillede": 2097,
"28": 2092,
"picture 13": 2092,
"item image 1": 2082,
"mã bảo mật": 2079,
"live camera from a random camera within the united states": 2079,
"slika": 2073,
"print": 2062,
"agnes monica - rindu | official video": 2060,
"qr-code": 2058,
"click here for full size photo!": 2042,
"background image": 2040,
"view image": 2034,
"view": 2030,
"loading...": 2029,
"client": 2026,
"image default": 2022,
"no caption": 2019,
"c": 2007,
"capture": 2006,
"user's avatar": 1996,
"https%3a%2f%2fimages": 1992,
"forsideomslag": 1990,
"image.jpeg": 1987,
"передня обкладинка": 1986,
"游民星空": 1977,
"pinterest": 1976,
"カスタマー画像": 1974,
"8": 1974,
"frente": 1965,
"29": 1963,
"listing image 38": 1962,
"featured": 1961,
"quote": 1959,
"market position of the selected technologies": 1954,
"sampul depan": 1952,
"εξώφυλλο": 1952,
"mastercam": 1952,
"イメージ 1": 1950,
"listing image 37": 1933,
"3": 1931,
"foto do imóvel": 1924,
"外観写真": 1922,
"?": 1920,
"graphisoft: archicad download 30-day free trial": 1909,
"table 2": 1907,
"h": 1895,
"bfmtv": 1888,
"asset-image": 1888,
"title page": 1885,
"kuva": 1884,
"view profile": 1874,
"13.jpg": 1873,
"show_image": 1872,
"erfgoedstuk": 1844,
"前表紙": 1843,
"artikelbild": 1843,
"ipb image": 1843,
"no gravatar": 1836,
"クリックで画像を表示": 1818,
"this is an image": 1806,
"foto di questa camera": 1806,
"objektbild": 1805,
"扫码下载豆瓣app": 1800,
"morepic-1": 1798,
"picture 14": 1788,
"圖片": 1779,
"play mp3": 1778,
"l": 1768,
"33": 1766,
"photo 11": 1766,
"loading": 1762,
"related image": 1760,
"ila_rendered": 1758,
"31": 1753,
"article preview thumbnail": 1752,
"chú thích ảnh": 1752,
"ブログ投稿画像": 1751,
"01.jpg": 1750,
"キッチン": 1749,
"listing image 39": 1747,
"predný obal": 1743,
"foto 2": 1733,
"4": 1732,
"blog": 1730,
"3.png": 1728,
"youtube preview image": 1726,
"9": 1724,
"imagegalleryimg": 1723,
"image thumb": 1719,
"b": 1717,
"サムネイル": 1712,
"portfolio-pic": 1708,
"商品画像1": 1700,
"32": 1698,
"cambia imagen": 1696,
"stockfoto": 1694,
"o": 1693,
"prüfcode": 1690,
"userpic": 1689,
"array": 1678,
"user avatar image": 1677,
"ürün görseli": 1675,
"p": 1671,
"%title插图%num": 1671,
"foto 3": 1669,
"tease photo": 1669,
"maxresdefault": 1667,
"գրքի շապիկի երեսը": 1665,
"this item is no longer available": 1659,
"img description": 1657,
"прикрепленное изображение": 1657,
"linkedin": 1653,
"kavak.com": 1651,
"image2": 1645,
"pointwise: reliable cfd meshing": 1641,
"14.jpg": 1638,
"morepic-2": 1637,
":)": 1633,
"story image": 1633,
"본문 사진": 1629,
"image 7": 1622,
"画像クリックでメニュー表示/非表示": 1616,
"slideshow image": 1611,
"please type the letters and numbers below": 1606,
"front view": 1605,
"photo 12": 1596,
"15.jpg": 1590,
"_.jpg": 1588,
"opište tento kontrolní kód": 1586,
"f": 1576,
"책제목": 1570,
"用户头像": 1561,
"commenter": 1560,
"blob.png": 1559,
"listing image 40": 1559,
"item image 2": 1558,
"view press release": 1557,
"view full size photo": 1548,
"larger view of product": 1542,
"center": 1538,
"5": 1536,
"g": 1524,
"34": 1524,
"youtube video": 1518,
"ปกหน้า": 1517,
"imgtoken": 1516,
"primary image": 1513,
"book cover image": 1512,
"přední strana obálky": 1508,
"qr-code dieser seite": 1498,
"article thumbnail": 1498,
"photo3.jpg": 1497,
"chart temporarily unavailable": 1491,
"image not available": 1489,
"썸네일 이미지": 1488,
"product img": 1483,
"enter the word": 1482,
"6": 1473,
"slider image": 1468,
"nächste seite": 1467,
"ホテル・敷地": 1464,
"cloudcast image": 1461,
"strada stefan cel mare - municipiul suceava": 1460,
"photo 13": 1460,
"next": 1452,
"picture 15": 1451,
"アイテムイメージ": 1449,
"โรงแรมและบริเวณโดยรอบ": 1448,
"profile-picture": 1446,
"点击查看大图": 1445,
"amazon": 1432,
"imagine": 1431,
"foto 4": 1430,
"16.jpg": 1423,
"введите код": 1422,
"presentation": 1421,
"slide 1": 1420,
"suceava (vedere de pe primăria suceava)": 1410,
"35": 1409,
"ön kapak": 1409,
"x": 1408,
"mania": 1407,
"???????????????????????????????": 1401,
"hotell og omgivelser": 1399,
"alternate image": 1399,
"illustration": 1399,
"飯店和廣場": 1398,
"img1": 1396,
"wallpaper-1689754": 1396,
"02.jpg": 1393,
"間取り画像": 1393,
"36": 1390,
"freetoedit": 1390,
"article image alt text": 1388,
"main picture": 1387,
"preview image": 1387,
"blue captcha image": 1384,
"ilustrační foto": 1380,
"7": 1378,
"登录查看大图": 1375,
"author gravatar": 1374,
"oda/süit": 1373,
"profilbild": 1362,
"slide image": 1360,
"ir a la comunidad": 1359,
"titolo libro": 1358,
"article feature image": 1356,
"クルマの写真": 1355,
"news-image": 1354,
"kiadvány fénykép": 1354,
"before": 1353,
"sprednja platnica": 1352,
"小紋": 1352,
"イメージ 2": 1349,
"图片关键词": 1348,
"ξενοδοχείο και εγκαταστάσεις": 1347,
"画像1": 1337,
"图片 1": 1336,
"nan": 1336,
"ahorro": 1336,
"after": 1334,
"rosegal": 1333,
"description": 1327,
"ms": 1324,
"нажмите на изображение, чтобы его изменить": 1312,
"featured product": 1312,
"[page image]": 1312,
"photo 14": 1310,
"image 8": 1304,
"legend": 1299,
"ซื้อ-ขาย บ้านมือสอง, คอนโดมือสอง": 1298,
"en": 1298,
"page 2": 1297,
"zdjęcie": 1297,
"thumb 1": 1296,
"profile pic": 1295,
"4.png": 1292,
"hotel i tereny wokół niego": 1291,
"الفندق والحدائق": 1289,
"17.jpg": 1284,
"figure 6": 1282,
"hqdefault": 1281,
"post-image": 1278,
"picture 16": 1275,
"cd-adapco": 1275,
"rum/svit": 1272,
"listing image 41": 1272,
"hqdefault1": 1271,
"view full image": 1270,
"resim": 1270,
"호텔 & 구내": 1269,
"padlet preview": 1268,
"habitación": 1263,
"TRUE": 1262,
"thumb image": 1260,
"カラーバリエーション": 1257,
"full-size item image": 1253,
"skudescription": 1251,
"rand3d": 1250,
"hotell och övriga anläggningar": 1250,
"<images posters": 1249,
"otel ve otel alanları": 1248,
"效果图": 1247,
"slide show": 1247,
"03.jpg": 1242,
"article featured image": 1241,
"자료 표지": 1238,
"foto 5": 1237,
"这里写图片描述": 1236,
"imagegallery": 1235,
"article picture": 1235,
"media": 1235,
"인사이트": 1234,
"hotel & perkemahan": 1233,
"отель и близлежащая территория": 1233,
"previous": 1233,
"18.jpg": 1231,
"slide3": 1231,
"slide2": 1230,
"εικόνα": 1227,
"10": 1226,
"룸/스위트": 1223,
"fotos deste quarto": 1220,
"hotel og udendørsarealer": 1219,
"item image 3": 1218,
"点击查看大图片": 1214,
"[ocr errors][ocr errors]": 1212,
"anonymous": 1211,
"8": 1209,
"photo_2": 1207,
"部屋・スイート": 1205,
"priceline": 1205,
"δωμάτιο/σουίτα": 1204,
"unknown": 1197,
"счетчик посещений counter.co.kz - бесплатный счетчик на любой вкус!": 1197,
"foto van deze kamer": 1196,
"ilustrační foto.": 1196,
"38": 1195,
"issue cover": 1193,
"{keywords}": 1193,
"微信分享": 1190,
"яндекс.метрика": 1190,
"klik for at se eller bestil foto": 1189,
"9": 1189,
"商品画像2": 1189,
"unable to view the image, please provide a valid url.": 1189,
"e": 1187,
"37": 1184,
"ห้องพัก/สวีท": 1183,
"photographie du bien": 1182,
"按此在新窗口浏览图片": 1182,
"غرفة/جناح": 1182,
"click here for full size original image": 1179,
"photo 15": 1177,
"listing image 42": 1177,
"immagine cliente": 1176,
"image3": 1175,
"44": 1173,
",": 1173,
"11": 1171,
"40": 1169,
"full size cover": 1166,
"pr_imgae": 1163,
"png": 1161,
"table 3": 1159,
"page 3": 1158,
"inquiry photo": 1157,
"見出し画像": 1155,
"thelocalsearchpros.com": 1154,
"colortrac by global scanning": 1152,
"suceava - panoramă zona centrală": 1152,
"vörumynd": 1151,
"image 9": 1148,
"slide4": 1146,
"about us": 1146,
"imagen del cliente": 1145,
"image-1": 1144,
"hotels.com": 1142,
"41": 1140,
"ilustračné foto": 1135,
"awaiting product image": 1134,
"房源图片": 1130,
"borrower image": 1130,
"메인사진": 1129,
"other": 1125,
"(bez tytułu)": 1125,
"listing main image": 1121,
"details": 1121,
"apartamento": 1120,
"listing image 43": 1119,
"無題": 1115,
"casa": 1113,
"a photo on flickr": 1113,
"썸네일": 1112,
"click the image to open in full size.": 1111,
"daily picdump": 1110,
"product-thumbnail": 1108,
"baño": 1108,
"contact us": 1106,
"sample image": 1105,
"42": 1103,
"divulgação": 1101,
"39": 1100,
"22.jpg": 1098,
"платье": 1097,
"youtube_profile_image": 1096,
"500": 1096,
"20.jpg": 1094,
"achat maison": 1093,
"k": 1093,
"漫画": 1092,
"t": 1091,
"new": 1089,
"quick view": 1089,
"epaper": 1088,
"member photo": 1088,
"thumb 81 8qiwqpal. sl1500": 1088,
"このページのqrコード": 1087,
"[merged small][ocr errors]": 1086,
"partner": 1086,
"pinterest_post": 1086,
"envíame un mensaje": 1085,
"客房/套房": 1084,
"ツイッターメディア": 1081,
"номер/люкс": 1081,
"back view": 1081,
"the": 1079,
"morepic-3": 1078,
"ungültiger sicherheitscode": 1078,
"123": 1076,
"post thumbnail": 1076,
"punjabkesari": 1075,
"q": 1075,
"índice": 1072,
"thumb 91kqayujzhl. sl1500": 1071,
"facebook_link": 1065,
"facebook_cover": 1064,
"board owner": 1064,
"gallery-image": 1063,
"0": 1063,
"イメージ 3": 1063,
"19.jpg": 1062,
"12": 1061,
"exif_jpeg_picture": 1060,
"이미지없음": 1059,
"photo4.jpg": 1057,
"111": 1056,
"instagram_story": 1052,
"photo 16": 1052,
"picture 17": 1051,
"twitter_post": 1050,
"instagram_post": 1046,
"illustration photo": 1046,
"5.png": 1046,
"21.jpg": 1045,
"玄関": 1045,
"listing image 44": 1045,
"<3": 1044,
"linkedin_post": 1043,
"view from room": 1043,
"รูปภาพ": 1043,
"[user picture]": 1041,
"quarto/suíte": 1038,
"コンサルタント": 1038,
"slide background": 1038,
"商品画像3": 1037,
"twitter_header": 1035,
"solidcam: program your cncs directly inside your existing cad system.": 1033,
"001.jpg": 1033,
"chambre/suite": 1032,
"04.jpg": 1031,
"slide show thumb": 1030,
"email_header": 1030,
"live help": 1029,
"image unavailable": 1029,
"sites-gant-dach-site": 1029,
"slide5": 1028,
"hôtel et jardin": 1028,
"pdf": 1027,
"foto de perfil": 1027,
"индекс цитирования": 1025,
"extension": 1024,
"zimmer/suite": 1024,
"mixiorek": 1023,
"page 4": 1022,
"slide1 n.": 1021,
"atrás": 1020,
"photo_print": 1017,
"ブラック": 1016,
"pirmais vāks": 1015,
"spotting image 1": 1014,
"mi foto": 1013,
"[ocr errors][merged small]": 1012,
"graphisoft archicad download a 30-day free trial": 1009,
"google photo": 1003,
"photo 17": 1002,
}

non_english_thre = {
        'ja': 0.99,
        'ar': 0.99,
        'el': 0.99,
        'ru': 0.99,
        'th': 0.99,
        'zh': 0.96,
        'uk': 0.99,
        'he': 0.99,
        'ko': 0.99,
        'fa': 0.99,
        'hy': 0.99,
        'bn': 0.99,
        'arz': 0.95,
        'bg': 0.98,
        'hi': 0.98,
        'mn': 0.95,
        'kn': 0.99,
        'pa': 0.98,
        'ta': 0.99,
        'ml': 0.95,
        'sr': 0.95,
        'te': 1,
        'mr': 0.97,
        'gu': 0.99,
        'eo': 0.98,
        'ka': 0.96,
        'km': 0.97,
        'ur': 0.95,
        'ne': 0.95,
        'am': 0.95,
        'yi': 0.95,
        'ky': 0.95,
        'kk': 0.97,
        'ckb': 0.95,
        'be': 0.99,
        'si': 0.99,
        'fy': 0.95,
        'mk': 0.96,
        'as': 0.99,
        'sd': 0.97,
        'ug': 0.96,
        'or': 0.96,
        'wuu': 0.96,
        'my': 1,
        'bo': 0.99,
        'ba': 0.95,
        'cv': 0.99,
        'ps': 0.95,
        'tt': 0.95,
        'tg': 0.95,
        'dv': 0.95,
        'yue': 0.98,
        'lo': 0.95,
        'ce': 0.97,
        'new': 0.95,
        }

mediaRegex = re.compile("\S*.jpg$|\S*.png$|\S*.jpeg$|\S*.svg$|\S*.mp4$")
prefixRegex = re.compile("img_\S|photo_\d*$|table \d*$|thumb \d*$")
f_idRegex = re.compile("f:id:\S*:\S*:\w*$")
urlRegex = re.compile("(?:http|ftp)s?://\S*$|www.\S*\.\S*$")
rsRegex = re.compile("rs=w:\d*,h:\d*\S*$")

start_list = [ # better to sort this list by word length
              'phone number',
              'error_codes',
              'no_photo',
              'wechat',
              'facebook',
              'wikidata',
              'youtube',
              '[image',
              'ngc',
              'unnamed',
              'thumbnail',
              'mls:',
              'picture',
              'id',
              'guest',
              'tumblr',
              'logo',
              'qr',
              'blog',
              'default',
              'image',
              'imag',
              'imgp',
              'img',
              'asset',
              'iso',
              'apple',
              'table',
              'item',
              'dscn',
              'dsc',
              '微信用户',
              '微信图片',
              'icon',
              'video',
              'user',
              'scan',
              'fc2',
              'ref',
              'photo',
              '\\frac',
              'isbn',
              'slide',
              'unknown',
              'fifa',
              'bnsf',
              'cimg',
              'page',
             ]
remove_list = [
               '.jpeg',
               '.jpg',
               '.png',
               '.gif',
              ]
end_list = [
            'datasheet',
            '@gmail.com',
            'icon',
            'thumbnail',
           ]
bad_whole_texts = [
             'cleanup image.jpg',
             'image.assetalttext',
             'window display',
             'usericon',
             'top',
             'windows 10 icon',
             '{{.firstitem.brand.name}}({{.firstitem.brand.namekana}})の「{{.firstitem.name}}」を使った{{.posttype}}',
             'list icon',
             'favicon icon',
             'menu',
             'tinyurl.com',
             'gmail',
             '房屋?#35745;?',
             'user_icon',
             'message icon',
             'videothumbnail',
             '<img src=',
             'linkedin-icon',
             'google.site',
             'flag',
             'image module',
             'header',
             'page-header',
             'alttext',
             'userimage',
             'instagram-icon',
             'tiktok',
             'pagetop',
             'floorplan.alttag',
             'header2',
             'user-avatar',
             'filtr icon',
             'users icon',
             'rss icon',
             'menu-icon',
             'secret code',
             'flag icon',
             'heading icon',
             'web template',
             'unused',
             'header_img',
            ]

def non_english_cleaning_0831(lang, score):
    if lang in non_english_thre and score >= non_english_thre[lang]:
        return False
    return True

def fasttext_cleaning(lang, score):
    if lang in nine_lang:
        if score > -1:
            return True
        else:
            return False
    else:
        if score > 0.2:
            return False
        else:
            return True


def bad_text_pattern_cleaning(data):
    data = data.lower()
    if bool(prefixRegex.match(data)):
        return False
    if bool(f_idRegex.match(data)):
        return False
    if bool(urlRegex.match(data)):
        return False
    if bool(rsRegex.match(data)):
        return False
    return True



def number_char_bad_text(x):
    tokens = re.split(' -_.')


def bad_pattern_data(x):
    x = x.lower()
    if x in bad_whole_texts:
        return False
    for item in remove_list:
        x = x.replace(item, '')
    for item in start_list:
        if x.startswith(item):
            x = x[len(item):]
            break
    for item in end_list:
        if x.endswith(item):
            x = x[:-len(item)]
            break
    tokens = re.split('\s|-|_|[.]|,', x)
    for token in tokens:
        if len(token) <= 1: # not a good token
            continue
        if token in ['ii', 'iii', 'iv', 'vi', 'vii', 'viii', 'ix']:
            continue
        pattern = []
        en_chars = 0
        for c in token:
            if c.isnumeric():
                pattern.append(0)
            else:
                pattern.append(1)
                if ord(c) >= 97 and ord(c) <= 122: # between 'a' and 'z'
                    en_chars += 1
        if en_chars <= 1:
            continue
        if en_chars <=2 and len(token) - en_chars >= 3: # sd345
            continue # bad
        pattern_diff = [abs(pattern[i]-pattern[i-1]) for i in range(1,len(pattern))]
        if sum(pattern_diff) >= 2 and len(token) >= 4: # e4yf, 4a68
            continue
        return True

    return False
