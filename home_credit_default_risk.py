# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-23T17:22:43.915864Z","iopub.execute_input":"2021-08-23T17:22:43.916352Z","iopub.status.idle":"2021-08-23T17:23:01.884285Z","shell.execute_reply.started":"2021-08-23T17:22:43.916252Z","shell.execute_reply":"2021-08-23T17:23:01.881581Z"}}
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from home_credit_utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # FEATURE_1: NAME_TYPE_SUITE

    df.loc[(df["NAME_TYPE_SUITE"] == "Other_B"), "NAME_TYPE_SUITE"] = 1
    df.loc[(df["NAME_TYPE_SUITE"] == "Other_A"), "NAME_TYPE_SUITE"] = 2
    df.loc[(df["NAME_TYPE_SUITE"] == "Group of people"), "NAME_TYPE_SUITE"] = 2
    df.loc[(df["NAME_TYPE_SUITE"] == "Unaccompanied"), "NAME_TYPE_SUITE"] = 2
    df.loc[(df["NAME_TYPE_SUITE"] == "Spouse, partner"), "NAME_TYPE_SUITE"] = 3
    df.loc[(df["NAME_TYPE_SUITE"] == "Children"), "NAME_TYPE_SUITE"] = 3
    df.loc[(df["NAME_TYPE_SUITE"] == "Family"), "NAME_TYPE_SUITE"] = 3

    df["NAME_TYPE_SUITE"].unique()

    # FEATURE_2: NAME_EDUCATION_TYPE

    edu_map = {'Lower secondary': 1,
               'Secondary / secondary special': 2,
               'Incomplete higher': 2,
               'Higher education': 3,
               'Academic degree': 4}

    df["NAME_EDUCATION_TYPE"].unique()

    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(edu_map).astype('int')

    # FEATURE_3: NAME_FAMILY_STATUS

    fam_map = {'Civil marriage': 1,
               'Single / not married': 1,
               'Separated': 2,
               'Married': 3,
               'Widow': 4,
               'Unknown': 5}

    df["NAME_FAMILY_STATUS"].unique()

    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].map(fam_map).astype('int')

    # FEATURE_4: NAME_HOUSING_TYPE

    df.loc[(df["NAME_HOUSING_TYPE"] == "Rented apartment"), "NAME_HOUSING_TYPE"] = 1
    df.loc[(df["NAME_HOUSING_TYPE"] == "With parents"), "NAME_HOUSING_TYPE"] = 2
    df.loc[(df["NAME_HOUSING_TYPE"] == "Municipal apartment"), "NAME_HOUSING_TYPE"] = 3
    df.loc[(df["NAME_HOUSING_TYPE"] == "Co-op apartment"), "NAME_HOUSING_TYPE"] = 4
    df.loc[(df["NAME_HOUSING_TYPE"] == "House / apartment"), "NAME_HOUSING_TYPE"] = 4
    df.loc[(df["NAME_HOUSING_TYPE"] == "Office apartment"), "NAME_HOUSING_TYPE"] = 5

    df["NAME_HOUSING_TYPE"].unique()

    df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].astype('int')

    # FEATURE_5: OCCUPATION_TYPE

    df.loc[(df["OCCUPATION_TYPE"] == "Low-skill Laborers"), "OCCUPATION_TYPE"] = 1
    df.loc[(df["OCCUPATION_TYPE"] == "Drivers"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Waiters/barmen staff"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Security staff"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Laborers"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Cooking staff"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Sales staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Cleaning staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Realty agents"), "OCCUPATION_TYPE"] = 5
    df.loc[(df["OCCUPATION_TYPE"] == "Secretaries"), "OCCUPATION_TYPE"] = 6
    df.loc[(df["OCCUPATION_TYPE"] == "Medicine staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "Private service staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "IT staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "HR staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "Core staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "High skill tech staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "Managers"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "Accountants"), "OCCUPATION_TYPE"] = 8

    df["OCCUPATION_TYPE"].unique()

    # FEATURE_6: REGION_RATING_CLIENT

    rate_map = {3: 1,
                2: 2,
                1: 3}

    df['REGION_RATING_CLIENT'] = df['REGION_RATING_CLIENT'].map(rate_map)

    df["REGION_RATING_CLIENT"].unique()

    # FEATURE_7: Kişinin başvurudan önceki çalışma gün sayısının kişinin yaşının gün cinsine oranı
    df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # FEATURE_8: Müşterinin gelirinin kredi tutarına oranı
    df['NEW_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

    # FEATURE_9: Müşterinin gelirinin aile üyesi miktarına oranı (Kişi başı gelir)
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # FEATURE_10: Kredi yıllık ödemesinin müşterinin gelirine oranı
    df['NEW_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # FEATURE_11: Kredi yıllık ödemesinin toplam kredi tutarına oranı
    df['NEW_PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # FEATURE_12: Toplam kredi tutarının malların tutarına oranı
    df["NEW_GOODS_RATE"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

    # FEATURE_13: Malın fiyatını geçen kredi analizi
    df['NEW_OVER_EXPECT_CREDIT'] = (df.AMT_CREDIT > df.AMT_GOODS_PRICE).replace({False: 0, True: 1})

    # FEATURE_14: Müşterinin gelirinin çocuk sayısına oranı (Saçma, düşükse sil)
    df['NEW_INCOME_FOR_CHILD_RATE'] = [x / y if y != 0 else 0 for x, y in
                                       df[['AMT_INCOME_TOTAL', 'CNT_CHILDREN']].values]

    # FEATURE_15: Aile Üyeleri - Çocuk Sayısı = Yetişkin Sayısı
    df["NEW_CNT_ADULTS"] = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]

    # FEATURE_16: Çocuk sayısının aile üyesi miktarına oranı
    df['NEW_CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    # FEATURE_17: Müşterinin yaşı (yıl cinsinden)
    df['NEW_DAYS_BIRTH'] = df['DAYS_BIRTH'] * -1 / 365
    df["NEW_YEARS_BIRTH"] = round(df['NEW_DAYS_BIRTH'], 0)

    # FEATURE_18: Sağlanan belgelerin toplamı
    doc_list = ["FLAG_DOCUMENT_3", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6",
                "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_11",
                "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
                "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19"]

    df["NEW_SUM_DOCUMENTS"] = df[doc_list].sum(axis=1)

    # FEATURE_19: Hafta içi - hafta sonu sınıflandırması
    df["NEW_DAY_APPR_PROCESS_START"] = "Weekdays"
    df["NEW_DAY_APPR_PROCESS_START"][
        (df["WEEKDAY_APPR_PROCESS_START"] == "SATURDAY") | (df["WEEKDAY_APPR_PROCESS_START"] == "SUNDAY")] = "Weekend"

    # df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)  # Araba yaş değeri boş olan gözlemler 0 olarak atandı

    # FEATURE_20: Müşterinin arabasının yaşının müşterinin günlük yaşına oranı
    df["NEW_OWN_CAR_AGE_PERC"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]

    # FEATURE_21: Kişinin araba yaşının çalıştığı yıla oranı
    df['NEW_CAR_EMPLOYED_PERC'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']

    # FEATURE_22: Müşterinin yaşınınmüşterinin arabasının yaşına oranı
    df["NEW_YEARS_CAR_PERC"] = df["NEW_YEARS_BIRTH"] / df["OWN_CAR_AGE"]

    # df["DAYS_ID_PUBLISHED_RATIO"] = df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"]
    # df["DAYS_REGISTRATION_RATIO"] = df["DAYS_REGISTRATION"] / df["DAYS_BIRTH"]
    # df["DAYS_LAST_PHONE_CHANGE_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]

    # FEATURE_23: Müşterinin yaşadığı yerin ortalama değerlendirmesinin toplamı
    living_avg_list = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', \
                       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', \
                       'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', \
                       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']

    df["NEW_SUM_LIVING_AVG"] = df[living_avg_list].sum(axis=1)

    # FEATURE_24: Müşterinin yaşadığı yerin mode değerlendirmesinin toplamı
    living_mode_list = ['APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                        'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
                        'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
                        'NONLIVINGAREA_MODE', 'TOTALAREA_MODE']

    df["NEW_SUM_LIVING_MODE"] = df[living_mode_list].sum(axis=1)

    # FEATURE_25: Dışarıdan alınan normalleştirilmiş puanların ortalaması
    df['NEW_EXT_SOURCE_MEAN'] = (df['EXT_SOURCE_1'] +
                                 df['EXT_SOURCE_2'] +
                                 df['EXT_SOURCE_3']) / 3

    # FEATURE_26: Dışarıdan alınan normalleştirilmiş puanların çarpımı
    df['NEW_EXT_SOURCE_MUL'] = df['EXT_SOURCE_1'] * \
                               df['EXT_SOURCE_2'] * \
                               df['EXT_SOURCE_3']

    # FEATURE_27: Dışarıdan alınan normalleştirilmiş puanların varyansı
    df['NEW_EX3_SOURCE_VAR'] = [np.var([ext1, ext2, ext3]) for ext1, ext2, ext3 in
                                zip(df['EXT_SOURCE_1'], df['EXT_SOURCE_2'], df['EXT_SOURCE_3'])]
    # REGION_RATING_CLIENT_W_CITY

    # DEF_60_CNT_SOCIAL_CIRCLE

    ##########################################################################

    # Generating new fetures with using other features

    # FEATURE_28: Dışarıdan alınan normalleştirilmiş puanların toplamı
    df['EXT_SOURCE_SUM'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)

    # FEATURE_29: Kişinin yıl cinsinden yaşının kredi tutarına oranı
    df['BIRTH_VS_CREDIT'] = [(a / b) if b != 0 else 0 for a, b in df[['NEW_YEARS_BIRTH', 'AMT_CREDIT']].values]

    # FEATURE_30: Müşterinin yıl cinsinden yaşının maaşına oranı (Müşterinin yaşam kalitesi) (sorun yaratırsa positive yap)
    df['NEW_BIRTH_INCOME_PERC'] = [(a / b) if b != 0 else 0 for a, b in
                                   df[['NEW_YEARS_BIRTH', 'AMT_INCOME_TOTAL']].values]

    # FEATURE_31: OBS TOTAL
    df["NEW_OBS_30_60"] = df[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']].sum(axis=1)

    # FEATURE_32: DEF TOTAL
    df["NEW_DEF_30_60"] = df[['DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']].sum(axis=1)

    # FEATURE_33: CITY ADRES UYUŞMAZLIKLARI (DAYS_ID_PUBLISH İLE) (DAYS_REGISTRATION İLE DENE)
    df["NEW_CHEAT_CITY"] = df["REG_CITY_NOT_LIVE_CITY"] + \
                           df["REG_CITY_NOT_WORK_CITY"] + \
                           df["LIVE_CITY_NOT_WORK_CITY"]

    df["NEW_CHEAT_CITY_EQ"] = (df["NEW_CHEAT_CITY"] + 1) * \
                              (df["DAYS_ID_PUBLISH"])

    df.drop("NEW_CHEAT_CITY", axis=1, inplace=True)

    # FEATURE_34: REGION ADRES UYUŞMAZLIKLARI (DAYS_ID_PUBLISH İLE) (DAYS_REGISTRATION İLE DENE)
    df["NEW_CHEAT_REGION"] = df["REG_REGION_NOT_LIVE_REGION"] + df["REG_REGION_NOT_WORK_REGION"] + df[
        "LIVE_REGION_NOT_WORK_REGION"]

    df["NEW_CHEAT_REGION_EQ"] = (df["NEW_CHEAT_REGION"] + 1) * \
                                (df["DAYS_ID_PUBLISH"])

    df.drop("NEW_CHEAT_REGION", axis=1, inplace=True)

    # FEATURE_35: Müşterinin yaşadığı bölge için ve müşterinin yaşadığı şehir için puanların çarpımı
    df["NEW_RATING_CLIENT"] = df["REGION_RATING_CLIENT"] * \
                              df["REGION_RATING_CLIENT_W_CITY"]

    # FEATURE_36: Müşterinin yaşadığı bölge ve şehir puanlamasının müşterinin geliri ile çarpılması
    df['NEW_RATING_CLIENT_INCOME'] = (df['REGION_RATING_CLIENT'] + df['REGION_RATING_CLIENT_W_CITY']) * df[
        'AMT_INCOME_TOTAL']

    # FEATURE_37: Kredi tutarının aile yetişkinleri sayısına bölünmesi
    df['NEW_AMT/FAM'] = df['AMT_CREDIT'] / df["NEW_CNT_ADULTS"]

    # FEATURE_38: Kişinin aylık gelir hesabı
    df['NEW_INCOME_IN_A_MONTH'] = df['AMT_INCOME_TOTAL'] / 12

    # FEATURE_38: Kişinin aylık kredi ödeme hesabı
    df['NEW_AMT_ANNUITY_IN_A_MONTH'] = df['AMT_ANNUITY'] / 12

    # FEATURE_39: Kişinin aylık gelirinden, aylık kredi ödeme tutarının çıkarılması (Aylık cepte kalan para)
    df['NEW_MONEY_MONTH'] = df['NEW_INCOME_IN_A_MONTH'] - df['NEW_AMT_ANNUITY_IN_A_MONTH']

    # FEATURE_40: Kredi ödemesinin kaç yılda biteceğinin hesabı
    df["NEW_HOW_MANY_YEARS_CREDIT"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    # FEATURE_41: Kredi tutarının malin fiyatına göre fazlalığı
    df["NEW_CREDIT_GOODS_SUBSTRACT"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]

    # FEATURE_42: Malın fiyatının kredi tutarına göre fazlalığı
    df["NEW_GOODS_CREDIT_SUBSTRACT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]

    # FEATURE_43: Cepte kalan paranın kişi başına düşen miktarı
    df['NEW_MONEY_MONTH_PER_PERSON'] = df['NEW_MONEY_MONTH'] / df['CNT_FAM_MEMBERS']

    # FEATURE_44: Eğitim kırılımı bazında maaş değişkeni oluşturulması
    NEW_INC_EDU = df[['AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE']].groupby('NAME_EDUCATION_TYPE').median()[
        'AMT_INCOME_TOTAL']

    df['NEW_INC_EDU'] = df['NAME_EDUCATION_TYPE'].map(NEW_INC_EDU)

    # FEATURE_45: Meslek kırılımı bazında maaş değişkeni oluşturulması
    NEW_INC_ORG = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()[
        'AMT_INCOME_TOTAL']

    df['NEW_INC_ORG'] = df['ORGANIZATION_TYPE'].map(NEW_INC_ORG)

    # FEATURE_46: Sorguların toplamı
    df['NEW_REQ_CREDIT_BUREAU_SUM'] = df[['AMT_REQ_CREDIT_BUREAU_DAY',
                                          'AMT_REQ_CREDIT_BUREAU_HOUR',
                                          'AMT_REQ_CREDIT_BUREAU_WEEK',
                                          'AMT_REQ_CREDIT_BUREAU_MON',
                                          'AMT_REQ_CREDIT_BUREAU_QRT',
                                          'AMT_REQ_CREDIT_BUREAU_YEAR']].sum(axis=1)

    # FEATURE_47: OBS'de 30 gün ve 60 gün arasına düşen kişiler
    df['NEW_OBS_30_OBS_60_BETWEEN'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] - df['OBS_60_CNT_SOCIAL_CIRCLE']

    # FEATURE_48: DEF'de 30 gün ve 60 gün arasına düşen kişiler
    df['NEW_DEF_30_DEF_60_BETWEEN'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] - df['DEF_60_CNT_SOCIAL_CIRCLE']

    # FEATURE_49: Mal fiyatının evin alan ortalamasına bölünmesi (metrekare başına düşen ücret)
    df['RATIO_AMT_GOODS_PRICE_TO_LIVINGAREA_AVG'] = df['AMT_GOODS_PRICE'] / df['LIVINGAREA_AVG']

    # FEATURE_50: Mal fiyatının binanın yaş ortalamasına bölünmesi
    df['RATIO_AMT_GOODS_PRICE_TO_YEARS_BUILD_AVG'] = df['AMT_GOODS_PRICE'] / df['YEARS_BUILD_AVG']

    # FEATURE_51: Mal fiyatının yaşanılan apartmanın durumuna bölünmesi
    df['RATIO_AMT_GOODS_PRICE_TO_LIVINGAPARTMENTS_AVG'] = df['AMT_GOODS_PRICE'] / df['LIVINGAPARTMENTS_AVG']

    # FEATURE_52: Kaç gün önce telefon değiştirme bilgisinin kaç gün önce kayıt değiştirme bilgisine oranı
    df['RATIO_DAYS_LAST_PHONE_CHANGE_TO_DAYS_REGISTRATION'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_REGISTRATION']

    # FEATURE_53: 1 saat önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_HOUR'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_54: 1 gün önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_DAY'] = df['AMT_REQ_CREDIT_BUREAU_DAY'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_55: 1 hafta önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_WEEK'] = df['AMT_REQ_CREDIT_BUREAU_WEEK'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_56: 1 ay önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_MON'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_57: 3 ay önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_QRT'] = df['AMT_REQ_CREDIT_BUREAU_QRT'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_58: 1 yıl önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_YEAR'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_59: Normalleştirilmiş puanın max'ı
    df['NEW_EXT_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)

    # FEATURE_60: Normalleştirilmiş puanın min'ı
    df['NEW_EXT_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)

    # FEATURE_61: Flag değişkenlerinin toplanması
    df['NEW_FLAG_CONTACTS_SUM'] = df['FLAG_MOBIL'] + \
                                  df['FLAG_EMP_PHONE'] + \
                                  df['FLAG_WORK_PHONE'] + \
                                  df['FLAG_CONT_MOBILE'] + \
                                  df['FLAG_PHONE'] + \
                                  df['FLAG_EMAIL']

    drop_list_1 = ["NONLIVINGAPARTMENTS_AVG", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAPARTMENTS_MEDI",
                   "AMT_REQ_CREDIT_BUREAU_QRT", "NEW_INCOME_CREDIT_PERC", "NEW_RATING_CLIENT_INCOME",
                   "NEW_INCOME_IN_A_MONTH", "NEW_MONEY_MONTH", "NEW_REQ_CREDIT_BUREAU_SUM",
                   "RATIO_AMT_GOODS_PRICE_TO_YEARS_BUILD_AVG", "RATIO_DAYS_LAST_PHONE_CHANGE_TO_DAYS_REGISTRATION",
                   "PERC_ENQUIRIES_HOUR", "PERC_ENQUIRIES_DAY", "PERC_ENQUIRIES_WEEK"]

    df.drop(drop_list_1, axis=1, inplace=True)

    drop_list_2 = ["APARTMENTS_MODE", "LIVINGAPARTMENTS_MEDI", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_AVG",
                   "LIVINGAREA_MEDI", "BASEMENTAREA_MEDI", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MEDI",
                   "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MEDI", "YEARS_BUILD_MODE", "COMMONAREA_MEDI",
                   "COMMONAREA_MODE", "ELEVATORS_MEDI", "ELEVATORS_MODE", "ENTRANCES_MEDI", "ENTRANCES_MODE",
                   "FLOORSMAX_MEDI", "FLOORSMAX_MODE", "FLOORSMIN_MEDI", "FLOORSMIN_MODE", "LANDAREA_MEDI",
                   "LANDAREA_MODE", "APARTMENTS_AVG", "NONLIVINGAREA_MEDI", "NONLIVINGAREA_MODE", "BASEMENTAREA_MEDI",
                   "BASEMENTAREA_AVG", "YEARS_BUILD_MEDI", "YEARS_BUILD_AVG", "ENTRANCES_MEDI", "ENTRANCES_AVG"]

    df.drop(drop_list_2, axis=1, inplace=True)

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    df.columns = pd.Index(["APP_" + col for col in df.columns.tolist()])
    df.rename(columns={"APP_SK_ID_CURR": "SK_ID_CURR"}, inplace=True)
    df.rename(columns={"APP_TARGET": "TARGET"}, inplace=True)

    del test_df
    gc.collect()
    return df


def bureau_and_balance(num_rows=None, nan_as_category=True):
    br = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
    bb = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(br)

    # XNA bulunduran yok.
    XNA_list = []
    for col in cat_cols:
        for i in range(len(br[col].unique())):
            if br[col].unique()[i] == "XNA":
                XNA_list.append(col)

    print(XNA_list)

    # br[num_cols].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.99]).T

    # for col in cat_cols:
    #     cat_summary(br, col)

    # CORRELATION
    # high_correlation(br)

    # RARE ENCODER
    br.loc[br["CREDIT_ACTIVE"] == "Sold", "CREDIT_ACTIVE"] = "Closed"
    br.loc[br["CREDIT_ACTIVE"] == "Bad debt", "CREDIT_ACTIVE"] = "Closed"

    br.drop("CREDIT_CURRENCY", axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(br)

    br = rare_encoder(br, 0.01)

    # rare_analyser_new(br, cat_cols)

    # FEATURE ENGINEERING

    # FEATURE: Kredi başvurusu bilgilerinin ne kadar sürede geldiği (Örn 188 gün içinde geldi)
    br['NEW_DAYS_CREDIT_UPDATE_SUBSTRACT'] = br['DAYS_CREDIT'] - br['DAYS_CREDIT_UPDATE']

    # FEATURE: Kişi kaç gün erken veya kaç gün geç ödemiş (kredinin kapanması gereken zaman - kredinin kapandığı zaman)
    br["NEW_CREDIT_PAID_SUBSTRACT"] = br["DAYS_CREDIT_ENDDATE"] - br["DAYS_ENDDATE_FACT"]

    # FEATURE: Kredinin kapandığı zaman <= Kredinin bitmesine ne kadar kaldığı (yarın hocaya sorulacak)
    # br['NEW_EARLY_PAID'] = (br['DAYS_ENDDATE_FACT'] <= br['DAYS_CREDIT_ENDDATE']).astype('float')
    # Geçliği ve erkenliği ayrı ayrı ifade eden kod satırları:
    br["NEW_LATE"] = br["NEW_CREDIT_PAID_SUBSTRACT"].apply(lambda x: 1 if x < 0 else 0)  # Gecikme olup olmaması
    br["NEW_EARLY"] = br["NEW_CREDIT_PAID_SUBSTRACT"].apply(lambda x: 1 if x > 0 else 0)  # Erken ödenip ödenmeme

    # FEATURE: Geciken kredi tutarının toplam kredi tutarına oranı
    br['NEW_OVERDUE_CREDIT_SUM_PERC'] = br['AMT_CREDIT_SUM_OVERDUE'] / br['AMT_CREDIT_SUM']

    # FEATURE: Mevcut borcun tüm kredi tutarına oranı
    br['NEW_DEBT_SUM_TO_CREDIT_SUM_RATIO'] = br['AMT_CREDIT_SUM_DEBT'] / br[
        'AMT_CREDIT_SUM']  # bunu bahara söyle amt_credit_sum +1 yapan var.

    # FEATURE: Mevcut kredi mikarı - Mevcut borç = Ödenen kredi miktarı
    br["NEW_PAID_CREDIT"] = br["AMT_CREDIT_SUM"] - br["AMT_CREDIT_SUM_DEBT"]

    # FEATURE: Mevcut kredi miktarı / mevcut borç
    br['NEW_DEBT_CREDIT_RATIO'] = br['AMT_CREDIT_SUM'] / br['AMT_CREDIT_SUM_DEBT']

    # FEATURE: Ödenen kredi miktarının yüzdesi
    br["NEW_PAID_CREDIT_PERC"] = (br["NEW_PAID_CREDIT"] / br["AMT_CREDIT_SUM"]) * 100

    # FEATURE: Gecikmiş tutarın uzatma miktarına oranı
    br['NEW_CREDIT_OVERDUE_PROLONG_SUM'] = [x / y if y != 0 else 0 for x, y in
                                            br[['AMT_CREDIT_SUM_OVERDUE', 'CNT_CREDIT_PROLONG']].values]

    # FEATURE: Maksimum gecikme tutarının uzatma miktarına oranı
    br['NEW_CREDIT_OVERDUE_PROLONG_MAX'] = [x / y if y != 0 else 0 for x, y in
                                            br[['AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG']].values]

    # FEATURE: Geciktirmeden ödediği tutar
    br['NEW_CREDIT_OVERDUE_SUBSTRACT'] = br['AMT_CREDIT_SUM'] - br['AMT_CREDIT_SUM_OVERDUE']

    # FEATURE: Kredi tutarının gecikme tutarına oranı
    br['NEW_CREDIT_OVERDUE_RATIO'] = br['AMT_CREDIT_SUM'] / br['AMT_CREDIT_SUM_OVERDUE']

    # FEATURE: Kredi Bürosunda bildirilen kredi kartının mevcut limiti - Kredi Bürosu kredisinde mevcut borç
    # br['NEW_AMT_CREDIT_DEBT_SUBSTRACT'] = br['AMT_CREDIT_SUM_LIMIT'] - br['AMT_CREDIT_SUM_DEBT']

    # FEATURE:
    # br['NEW_AMT_CREDIT_DEBT_RATIO'] = br['AMT_CREDIT_SUM_DEBT'] / br['AMT_CREDIT_SUM_LIMIT']

    # FEATURE:
    br["NEW_HAS_CREDIT_CARD"] = br["AMT_CREDIT_SUM_LIMIT"].apply(lambda x: 1 if x > 0 else 0)

    # Aylık Ödeme Oranı
    br['NEW_AMT_ANNUITY_RATİO'] = br['AMT_ANNUITY'] / br['AMT_CREDIT_SUM']  # bunu sonradan ekledim kesinlikle bak .

    # FEATURE:
    br.loc[br['CREDIT_ACTIVE'] == "Closed", "NEW_IS_ACTIVE_CREDIT"] = 0
    br.loc[br['CREDIT_ACTIVE'] == "Active", "NEW_IS_ACTIVE_CREDIT"] = 1
    br["NEW_IS_ACTIVE_CREDIT"] = br["NEW_IS_ACTIVE_CREDIT"].astype("int")

    # FEATURE: Müşterinin şimdiye kadar yaptığı toplam kredi başvuru sayısı (olumlu olumsuz başvurular dahil)
    group = br[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(
        index=str, columns={'DAYS_CREDIT': 'NEW_BUREAU_LOAN_COUNT'})
    br = br.merge(group, on=['SK_ID_CURR'], how='left')  # ana tablo ile birleştirme.

    # FEATURE: Müşterinin şimdiye kadar yaptığı kredi başvurularının türünün sayısı
    group = br[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(
        index=str, columns={'CREDIT_TYPE': 'NEW_BUREAU_LOAN_TYPES'})
    br = br.merge(group, on=['SK_ID_CURR'], how='left')

    # FEATURE: Müşterinin kredi türü başına düşen başvuru sayısı. "Müşteri farklı türlerde kredi almış mı, yoksa tek bir çeşit kredi mi kullanmış", bunu gözlemliyoruz.
    br['NEW_AVERAGE_LOAN_TYPE'] = br['NEW_BUREAU_LOAN_COUNT'] / br['NEW_BUREAU_LOAN_TYPES']

    # FEATURE: Müşteri başına aktif kredilerin ortalama sayısı
    # df['CREDIT_ACTIVE_BINARY'] = df['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Active' else 0)

    br.loc[br['CREDIT_ACTIVE'] == "Closed", 'CREDIT_ACTIVE_BINARY'] = 0
    br.loc[br['CREDIT_ACTIVE'] != "Closed", 'CREDIT_ACTIVE_BINARY'] = 1
    br['CREDIT_ACTIVE_BINARY'] = br['CREDIT_ACTIVE_BINARY'].astype('int32')

    # Kapanmamış kredi borçları 1'e daha yakın ise bu iyi değildir.
    group = br.groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ACTIVE_BINARY': 'NEW_ACTIVE_LOANS_PERCENTAGE'})
    br = br.merge(group, on=['SK_ID_CURR'], how='left')
    del br['CREDIT_ACTIVE_BINARY']
    gc.collect()

    # FEATURE: HER MÜŞTERİ İÇİN BAŞARILI GEÇMİŞ BAŞVURULAR ARASINDAKİ ORTALAMA GÜN SAYI
    # Müşteri geçmişte ne sıklıkla kredi aldı? Düzenli zaman aralıklarında mı dağıtıldı - iyi bir finansal planlamanın işareti mi yoksa krediler daha küçük bir zaman çerçevesi etrafında mı yoğunlaştı - potansiyel finansal sıkıntıyı mı gösteriyor?

    # Her Müşteriye göre gruplandırıldı ve DAYS_CREDIT değerleri artan düzende sıralandı.
    # Kredi DAYS_CREDIT'i SK_ID_CURR bazında sıralayarak NEW_DAYS_DIFF değişkeni üretmek kredi alma frekansı bilgisi verebilir.
    grp = br[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(drop=True)
    # rename(index = str, columns = {'DAYS_CREDIT': 'DAYS_CREDIT_DIFF'})
    print("Grouping and Sorting done")

    # Calculate Difference between the number of Days
    grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT'] * -1
    grp1['NEW_DAYS_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])[
        'DAYS_CREDIT1'].diff()  # aldığı farklı krediler arasında kaçar gün olduğu hesaplandı
    grp1['NEW_DAYS_DIFF'] = grp1['NEW_DAYS_DIFF'].fillna(0).astype(
        'uint32')  # ilk değişkende nan geleceği için 0 ile doldurdum. diff fonksiyonunda 2. değerden 1. değer çıkarılıyor. bu sebeple ilk değerde nan geliyor.
    del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT'], grp1['SK_ID_CURR']
    gc.collect()

    print("Difference days calculated")
    br = br.merge(grp1, on=['SK_ID_BUREAU'], how='left')
    print("Difference in Dates between Previous CB applications is CALCULATED")

    # Feature :Ödemesi devam eden kredi sayılarının ortalaması

    br.loc[br['DAYS_CREDIT_ENDDATE'] < 0, "CREDIT_ENDDATE_BINARY"] = 0  # ödemesi bitmiş (Closed) krediler
    br.loc[br['DAYS_CREDIT_ENDDATE'] >= 0, "CREDIT_ENDDATE_BINARY"] = 1  # ödemesi devam eden (Active) krediler
    grp = br.groupby(by=['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ENDDATE_BINARY': 'NEW_CREDIT_ENDDATE_PERCENTAGE'})

    br = br.merge(grp, on=['SK_ID_CURR'], how='left')
    del br['CREDIT_ENDDATE_BINARY']
    gc.collect()

    # FEATURE 7
    # AVERAGE NUMBER OF DAYS IN WHICH CREDIT EXPIRES IN FUTURE -INDICATION OF CUSTOMER DELINQUENCY IN FUTURE??
    # Repeating Feature 6 to Calculate all transactions with ENDATE as POSITIVE VALUES

    br['CREDIT_ENDDATE_BINARY'] = br['DAYS_CREDIT_ENDDATE']
    # Dummy column to calculate 1 or 0 values. 1 for Positive CREDIT_ENDDATE and 0 for Negat
    br.loc[(br["DAYS_CREDIT_ENDDATE"] <= 0), "CREDIT_ENDDATE_BINARY"] = 0  # ödemesi bitmiş (Closed) krediler
    br.loc[(br["DAYS_CREDIT_ENDDATE"] > 0), "CREDIT_ENDDATE_BINARY"] = 1  # ödemesi devam eden (Active) krediler
    print("New Binary Column calculated")

    # We take only positive values of  ENDDATE since we are looking at Bureau Credit VALID IN FUTURE
    # as of the date of the customer's loan application with Home Credit
    B1 = br[br['CREDIT_ENDDATE_BINARY'] == 1]
    del br["CREDIT_ENDDATE_BINARY"]

    # Calculate Difference in successive future end dates of CREDIT
    # Create Dummy Column for CREDIT_ENDDATE
    B1['DAYS_CREDIT_ENDDATE1'] = B1['DAYS_CREDIT_ENDDATE']
    # Groupby Each Customer ID
    grp = B1[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by=['SK_ID_CURR'])
    # Sort the values of CREDIT_ENDDATE for each customer ID
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending=True)).reset_index(drop=True)
    del grp
    gc.collect()
    print("Grouping and Sorting done")

    # Calculate the Difference in ENDDATES and fill missing values with zero
    grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff()
    grp1['NEW_DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
    del grp1['DAYS_CREDIT_ENDDATE1'], grp1['SK_ID_CURR']
    gc.collect()
    print("Difference days calculated")

    # Merge new feature 'DAYS_ENDDATE_DIFF' with original Data frame for BUREAU DATA
    br = br.merge(grp1, on=['SK_ID_BUREAU'], how='left')
    del grp1
    gc.collect()

    # FEATURE 8 - DEBT OVER CREDIT RATIO
    # The Ratio of Total Debt to Total Credit for each Customer
    # A High value may be a red flag indicative of potential default

    br['AMT_CREDIT_SUM_DEBT'] = br['AMT_CREDIT_SUM_DEBT'].fillna(0)
    br['AMT_CREDIT_SUM'] = br['AMT_CREDIT_SUM'].fillna(0)

    grp1 = br[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str,
                                                          columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})

    grp2 = br[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(
        index=str, columns={'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})

    br = br.merge(grp1, on=['SK_ID_CURR'], how='left')
    br = br.merge(grp2, on=['SK_ID_CURR'], how='left')
    del grp1, grp2
    gc.collect()

    br['NEW_DEBT_CREDIT_RATIO'] = br['TOTAL_CUSTOMER_DEBT'] / br['TOTAL_CUSTOMER_CREDIT']

    del br['TOTAL_CUSTOMER_DEBT'], br['TOTAL_CUSTOMER_CREDIT']
    gc.collect()

    # FEATURE 9 - OVERDUE OVER DEBT RATIO
    # What fraction of total Debt is overdue per customer?
    # A high value could indicate a potential DEFAULT

    br['AMT_CREDIT_SUM_DEBT'] = br['AMT_CREDIT_SUM_DEBT'].fillna(0)
    br['AMT_CREDIT_SUM_OVERDUE'] = br['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

    grp1 = br[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str,
                                                          columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = br[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(index=str, columns={
        'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})

    br = br.merge(grp1, on=['SK_ID_CURR'], how='left')
    br = br.merge(grp2, on=['SK_ID_CURR'], how='left')
    del grp1, grp2
    gc.collect()

    br['NEW_OVERDUE_DEBT_RATIO'] = br['TOTAL_CUSTOMER_OVERDUE'] / br['TOTAL_CUSTOMER_DEBT']

    del br['TOTAL_CUSTOMER_OVERDUE'], br['TOTAL_CUSTOMER_DEBT']
    gc.collect()

    # FEATURE 10 - AVERAGE NUMBER OF LOANS PROLONGED
    # Müşter
    br['CNT_CREDIT_PROLONG'] = br['CNT_CREDIT_PROLONG'].fillna(0)
    grp = br[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by=['SK_ID_CURR'])[
        'CNT_CREDIT_PROLONG'].mean().reset_index().rename(index=str, columns={
        'CNT_CREDIT_PROLONG': 'NEW_AVG_CREDITDAYS_PROLONGED'})
    br = br.merge(grp, on=['SK_ID_CURR'], how='left')

    # One Hot
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(br)

    # high_correlation(br, remove=['SK_ID_CURR','SK_ID_BUREAU'], corr_coef = "spearman", corr_value = 0.7)

    drop_list_1 = ["NEW_CREDIT_OVERDUE_SUBSTRACT", "NEW_DEBT_SUM_TO_CREDIT_SUM_RATIO", "NEW_PAID_CREDIT_PERC",
                   "NEW_HAS_CREDIT_CARD", "NEW_OVERDUE_CREDIT_SUM_PERC"]
    br.drop(drop_list_1, axis=1, inplace=True)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category=True)
    br, br_cat = one_hot_encoder(br, nan_as_category=True)

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(br)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    br = br.join(bb_agg, how='left', on='SK_ID_BUREAU')
    br.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'SK_ID_CURR': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['min', 'max'],
        'CNT_CREDIT_PROLONG': ['min', 'max'],
        'AMT_CREDIT_SUM': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'min'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean', 'min', 'sum'],
        'NEW_DAYS_CREDIT_UPDATE_SUBSTRACT': ['min', 'max', 'sum', 'mean'],
        'NEW_CREDIT_PAID_SUBSTRACT': ["count", "min", "max", "mean"],
        # 'NEW_OVERDUE_CREDIT_SUM_PERC': ['min', 'mean', 'max'],
        # 'NEW_DEBT_SUM_TO_CREDIT_SUM_RATIO': ['min', 'mean', 'max'],
        'NEW_PAID_CREDIT': ['sum'],
        'NEW_DEBT_CREDIT_RATIO': ['min', 'sum'],
        # 'NEW_PAID_CREDIT_PERC': ['mean'],
        'NEW_CREDIT_OVERDUE_PROLONG_SUM': ['mean'],
        'NEW_CREDIT_OVERDUE_PROLONG_MAX': ['mean', 'min'],
        # 'NEW_CREDIT_OVERDUE_SUBSTRACT': ['mean', 'min', 'max'],
        'NEW_CREDIT_OVERDUE_RATIO': ['min', 'max'],
        # 'NEW_BUREAU_LOAN_COUNT': ["min","max","sum","mean"],
        'NEW_AVERAGE_LOAN_TYPE': ["min", "max"],
        'NEW_BUREAU_LOAN_TYPES': ["min", "max", "mean", "sum"],
        'NEW_ACTIVE_LOANS_PERCENTAGE': ["min", "max"],
        'NEW_DAYS_DIFF': ["max", "mean"],
        'NEW_CREDIT_ENDDATE_PERCENTAGE': ["max"],
        'DAYS_ENDDATE_DIFF': ["min", "max", "mean"],
        'NEW_OVERDUE_DEBT_RATIO': ["max", "mean"],
        'NEW_EARLY': ["sum"],
        'NEW_LATE': ["sum"],
        # 'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']}

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in br_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = br.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # high_correlation(bureau_agg, remove=['SK_ID_CURR','SK_ID_BUREAU'], corr_coef = "spearman", corr_value = 0.7)

    # kisinin aldıgı en yuksek ve en dusuk kredinin farkını gösteren yeni degisken
    bureau_agg["BURO_NEW_AMT_CREDIT_SUM_RANGE"] = bureau_agg["BURO_AMT_CREDIT_SUM_MAX"] - bureau_agg[
        "BURO_AMT_CREDIT_SUM_MIN"]

    # # ortalama kac ayda bir kredi cektigini ifade eden  yeni degisken
    # bureau_agg["BURO_NEW_DAYS_CREDIT_RANGE"]= round((bureau_agg["BURO_DAYS_CREDIT_MAX"] - bureau_agg["BURO_DAYS_CREDIT_MIN"])/(30 * bureau_agg["BURO_SK_ID_CURR_COUNT"]))

    # NEW_EARLY_RATIO
    bureau_agg['NEW_EARLY_RATIO'] = bureau_agg['BURO_NEW_EARLY_SUM'] / bureau_agg[
        'BURO_NEW_CREDIT_PAID_SUBSTRACT_COUNT']  # Erken ödeme oranı

    # NEW_LATE_RATIO
    bureau_agg['NEW_LATE_RATIO'] = bureau_agg['BURO_NEW_LATE_SUM'] / bureau_agg[
        'BURO_NEW_CREDIT_PAID_SUBSTRACT_COUNT']  # Geç ödeme oranı

    # Bureau: Active credits - using only numerical aggregations
    active = br[br['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = br[br['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, br
    gc.collect()
    return bureau_agg


def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')

    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # prev[num_cols].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.99]).T

    # Numerikler için describe attığımda problemli 365243 değerinin alttaki değişkenlerde bulunduğunu gördüm.

    prev.loc[:, ['DAYS_FIRST_DRAWING',
                 'DAYS_FIRST_DUE',
                 'DAYS_LAST_DUE_1ST_VERSION',
                 'DAYS_LAST_DUE',
                 'DAYS_TERMINATION', ]].replace(365243.0, np.nan, inplace=True)

    # XNA sınıfını bulunduran değişkenleri gözlemlemek istiyorum:

    for col in cat_cols:
        print()
        print("*****************************")
        print("Classes of", col, ":")
        print(prev[col].unique())

    # # Sadece XNA bulunduran değişkenleri yazdır:
    # XNA_list = []
    # for col in cat_cols:
    #     for i in range(len(prev[col].unique())):
    #         if prev[col].unique()[i] == "XNA":
    #             XNA_list.append(col)
    # print(XNA_list)
    #
    # # XNA bulunduran değişkenleri np.nan ile değiş:
    # for col in cat_cols:
    #     for i in range(len(prev[col].unique())):
    #         if prev[col].unique()[i] == "XNA":
    #             prev[col].replace("XNA", np.nan, inplace=True)

    # CAT ANALYZER

    # for i in cat_cols + cat_but_car + num_but_cat:
    #     cat_analyzer(prev, i)

    drop_list = ['FLAG_LAST_APPL_PER_CONTRACT', 'NAME_TYPE_SUITE', 'NAME_SELLER_INDUSTRY', 'NFLAG_LAST_APPL_IN_DAY',
                 'NFLAG_LAST_APPL_IN_DAY', 'WEEKDAY_APPR_PROCESS_START']

    prev.drop(drop_list, axis=1, inplace=True)

    # Rare Encoder
    rare_cols = ["NAME_PAYMENT_TYPE", "CODE_REJECT_REASON", "CHANNEL_TYPE", "NAME_GOODS_CATEGORY",
                 "PRODUCT_COMBINATION"]

    for i in rare_cols:
        rare_encoder_prev(prev, i, rare_perc=0.01)

    prev["NAME_CASH_LOAN_PURPOSE"] = np.where(~prev["NAME_CASH_LOAN_PURPOSE"].isin(["XAP", "XNA"]), "Other",
                                              prev["NAME_CASH_LOAN_PURPOSE"])

    prev.loc[(prev["NAME_PORTFOLIO"] == "Cards"), "NAME_PORTFOLIO"] = "Cars"

    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # for i in cat_cols + cat_but_car + num_but_cat:
    #     cat_analyzer(prev, i)

    # FEATURE ENGINEERING

    # kategorik değişken kırılımında target analizi

    # cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # Feature Engineering

    # FEATURE: Müşterinin istediği kredi miktarının, müşterinin aldığı kredi miktarına oranı
    prev['NEW_APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # FEATURE:

    # 10/50 ---- 1 (İSTEDİĞİNDEN DE FAZLASINI ALDI --- 1 BASILIR)
    # 60/10 ---- 0 (İSTEDİĞİNDEN AZ ALDI --- 0 BASILIR)

    prev["NEW_APP_CREDIT_RATE_PERC"] = prev["NEW_APP_CREDIT_PERC"].apply(lambda x: 1 if (x <= 1) else 0)

    # FEATURE: Kredi peşinatının yıllık ödemeye oranı
    prev['NEW_AMT_PAYMENT_ANNUITY_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_ANNUITY']

    # FEATURE: Kredi peşinatının kredi tutarına oranı
    prev['NEW_AMT_PAYMENT_CREDIT_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']

    # FEATURE: Mal fiyatının kredi tutarına oranı
    prev['NEW_GOODS_PRICE_CREDIT_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']

    # FEATURE: Müşterinin talep ettiği miktardan, talep ettiği mal fiyatının çıkarılması
    prev['NEW_APPLICATION_GOODS_SUBSTRACT'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']

    # FEATURE: Müşterinin talep ettiği miktarın, talep ettiği mal fiyatına oranı
    prev['NEW_APPLICATION_GOODS_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_GOODS_PRICE']

    # FEATURE: Peşinat oranından faiz oranının çıkarılması
    prev['NEW_RATE_PAYMENT_INTEREST_PRIMARY'] = prev['RATE_DOWN_PAYMENT'] - prev['RATE_INTEREST_PRIMARY']

    # FEATURE:
    # prev['DIFF_RATE_INTEREST_PRIVILEGED_RATE_INTEREST_PRIMARY'] = prev['RATE_INTEREST_PRIVILEGED'] - prev['RATE_INTEREST_PRIMARY']

    # FEATURE: Son vade tarihinden ilk vade tarihinin çıkarılması
    prev['NEW_LAST_AND_FIRST_SUBSTRACT'] = prev['DAYS_LAST_DUE'] - prev['DAYS_FIRST_DUE']

    # FEATURE: Önceki başvurunun sonlandırılmasından, önceki başvuru için kararın ne zaman verildiğinin çıkarılması
    prev['NEW_TERMINATION_DECISION_SUBSTRACT'] = prev['DAYS_TERMINATION'] - prev['DAYS_DECISION']

    # FEATURE: Kredi tutarının vadeye oranı
    prev["NEW_CREDIT_PAYMENT_RATIO"] = prev["AMT_CREDIT"] / prev["CNT_PAYMENT"]

    # FEATURE: Aylık ödeme miktarının yıllık ödeme miktarına oranı
    prev["NEW_CREDIT_PAYMENT_YEAR_RATIO"] = prev["NEW_CREDIT_PAYMENT_RATIO"] / prev["AMT_ANNUITY"]

    # FEATURE: Kredi ödemesinin kaç yılda biteceğinin hesabı
    prev["NEW_CREDIT_TERM_YEAR"] = prev["CNT_PAYMENT"] / 12

    # FEATURE:
    prev["NEW_CNT_PAYMENT_CAT"] = pd.cut(x=prev['CNT_PAYMENT'], bins=[0, 12, 60, 120],
                                         labels=["Short", "Middle", "Long"])

    # FEATURE: Peşinat oranı ile peşinat miktarının çarpılması
    prev['NEW_RATE_AMT_DOWN_PAYMENT'] = prev['RATE_DOWN_PAYMENT'] * prev['AMT_DOWN_PAYMENT']

    # FEATURE: Kredi tutarının faiz oranı ile çarpılması (Genele uyarlanmış biçimde)
    # prev['NEW_CREDIT_PRIMARY_RATIO'] = prev['AMT_CREDIT'] * prev['RATE_INTEREST_PRIMARY'] (nan dolu, sildik)

    # FEATURE: Kredi tutarının faiz oranı ile çarpılması (Kişiye özel uyarlanmış biçimde)
    # prev['NEW_CREDIT_PRIVILEGED_RATIO'] = prev['AMT_CREDIT'] * prev['RATE_INTEREST_PRIVILEGED'] (nan dolu, sildik)

    # FEATURE: Mal fiyatının vadeye oranı (kişinin mal için ayda ne kadar ödediğini gözlemliyoruz)
    prev['NEW_GOODS_PAYMENT_PER_MONTH'] = prev['AMT_GOODS_PRICE'] / prev['CNT_PAYMENT']

    # FEATURE: Kredinin ödeneceği yıl miktarı ile yıllık kredi ödeme miktarının çarpılması
    prev['NEW_WHOLE_CREDIT'] = prev['NEW_CREDIT_TERM_YEAR'] * prev['AMT_ANNUITY']

    # FEATURE: Toplam ödenecek kredi miktarının, kredi tutarına oranı
    prev['NEW_WHOLE_CREDIT_AMT_CREDIT_RATIO'] = prev['NEW_WHOLE_CREDIT'] / (prev['AMT_CREDIT'])

    # FEATURE: "HOUR_APPR_PROCESS_START"  değişkeninin working_hours ve off_hours olarak iki kategoriye ayrılması
    work_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    prev["NEW_HOUR_APPR_PROCESS_START"] = prev["HOUR_APPR_PROCESS_START"].replace(work_hours, 'working_hours')

    off_hours = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
    prev["NEW_HOUR_APPR_PROCESS_START"] = prev["HOUR_APPR_PROCESS_START"].replace(off_hours, 'off_hours')

    # FEATURE: X-sell approved (karşı tarafın teklifini kabul edip kredi isteği kabul edilen müşteri)
    prev['NEW_X_SELL_APPROVED'] = 0
    prev.loc[(prev['NAME_PRODUCT_TYPE'] == 'x-sell') & (
                prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_X_SELL_APPROVED'] = 1

    # FEATURE: walk-in approved (kendi istediği kredi tutarını belirtip kabul edilen müşteri)
    prev['NEW_WALK_IN_APPROVED'] = 0
    prev.loc[(prev['NAME_PRODUCT_TYPE'] == 'walk-in') & (
                prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_WALK_IN_APPROVED'] = 1

    # FEATURE: Eski müşteri olup onaylanan müşteri
    prev['NEW_REPEATER_APPROVED'] = 0
    prev.loc[(prev['NAME_CLIENT_TYPE'] == 'Repeater') & (
                prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_REPEATER_APPROVED'] = 1

    # FEATURE: Yeni müşteri olup onaylanan müşteri
    prev['NEW_NEWCUST_APPROVED'] = 0
    prev.loc[
        (prev['NAME_CLIENT_TYPE'] == 'New') & (prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_NEWCUST_APPROVED'] = 1

    # FEATURE: Kayıt yenileyip onaylanan müşteri
    prev['NEW_REFRESHED_APPROVED'] = 0
    prev.loc[(prev['NAME_CLIENT_TYPE'] == 'Refreshed') & (
                prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_REFRESHED_APPROVED'] = 1

    # # FEATURE:
    # df_prev['NEW_HIGH_APPROVED'] = 0
    # df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'high') & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_HIGH_APPROVED'] = 1
    #
    # # FEATURE:
    # df_prev['NEW_MIDDLE_APPROVED'] = 0
    # df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'middle') & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_MIDDLE_APPROVED'] = 1
    #
    # # FEATURE:
    # df_prev['NEW_LOWACTION_APPROVED'] = 0
    # df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'low_action') & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_LOWACTION_APPROVED'] = 1
    #
    # # FEATURE:
    # df_prev['NEW_LOWNORMAL_APPROVED'] = 0
    # df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'low_normal') & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_LOWNORMAL_APPROVED'] = 1

    # credit requested / credit given ratio
    # df_prev['NEW_APP_CREDIT_RATIO'] = df_prev['AMT_APPLICATION'].div(df_prev['AMT_CREDIT']).replace(np.inf, 0) (Accuracy düşük çıkarsa buradaki yöntemi uygula)

    # prev['NEW_GOODS_PRICE_CREDIT_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']
    # # risk assessment via NEW_GOODS_PRICE_CREDIT_RATIO
    # prev.loc[prev['NEW_GOODS_PRICE_CREDIT_RATIO'] >= 1, 'NEW_CREDIT_GOODS_RISK'] = 0
    # prev.loc[prev['NEW_GOODS_PRICE_CREDIT_RATIO'] < 1, 'NEW_CREDIT_GOODS_RISK'] = 1

    # risk to approved
    # df_prev['NEW_RISK_APPROVED'] = 0
    # df_prev.loc[(df_prev['NEW_GOODS_PRICE_CREDIT_RATIO'] == 1) & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_RISK_APPROVED'] = 1

    # non risk to approved
    # df_prev['NEW_NONRISK_APPROVED'] = 0
    # df_prev.loc[(df_prev['NEW_CREDIT_GOODS_RISK'] == 0) & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_NONRISK_APPROVED'] = 1

    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # high_correlation(prev, remove=['SK_ID_CURR','SK_ID_PREV'], corr_coef = "spearman", corr_value = 0.7)

    drop_list1 = ["NEW_CREDIT_PAYMENT_RATIO", "NEW_RATE_AMT_DOWN_PAYMENT", "NEW_AMT_PAYMENT_CREDIT_RATIO",
                  "NEW_GOODS_PRICE_CREDIT_RATIO", "NEW_APPLICATION_GOODS_RATIO"]

    prev.drop(drop_list1, axis=1, inplace=True)

    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max'],
        'AMT_DOWN_PAYMENT': ['min', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'mean'],
        'HOUR_APPR_PROCESS_START': ['mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'NEW_APP_CREDIT_PERC': ['min', 'mean'],
        'NEW_AMT_PAYMENT_ANNUITY_RATIO': ['min', 'max'],
        # 'NEW_AMT_PAYMENT_CREDIT_RATIO': ['min', 'max', 'mean'],
        # 'NEW_GOODS_PRICE_CREDIT_RATIO': ['max', 'min'],
        'NEW_APPLICATION_GOODS_SUBSTRACT': ['min', 'max', 'mean'],
        # 'NEW_APPLICATION_GOODS_RATIO': ['mean'],
        'NEW_RATE_PAYMENT_INTEREST_PRIMARY': ['mean'],
        'NEW_LAST_AND_FIRST_SUBSTRACT': ['min', 'max', 'mean'],
        'NEW_TERMINATION_DECISION_SUBSTRACT': ['min', 'max', 'mean'],
        # 'NEW_CREDIT_PAYMENT_RATIO': ['min', 'max', 'mean'],
        'NEW_CREDIT_PAYMENT_YEAR_RATIO': ['min', 'max'],
        'NEW_CREDIT_TERM_YEAR': ['min', 'max', 'mean'],
        # 'NEW_RATE_AMT_DOWN_PAYMENT': ['sum', 'mean'],
        'NEW_GOODS_PAYMENT_PER_MONTH': ['sum', 'max', 'min'],
        'NEW_WHOLE_CREDIT': ['sum', 'mean'],
        'NEW_WHOLE_CREDIT_AMT_CREDIT_RATIO': ['mean']}

    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    # cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv', nrows=num_rows)
    pos["NAME_CONTRACT_STATUS"] = np.where(~(pos["NAME_CONTRACT_STATUS"].isin(['Active', 'Completed', 'Signed'])),"Rare", pos["NAME_CONTRACT_STATUS"])
    pos, cat_cols = one_hot_encoder(pos, nan_as_category)

    # Features
    aggregations = {'MONTHS_BALANCE': ['min', 'max'],
                    'CNT_INSTALMENT': ['min', 'max', 'std', 'median'],
                    'CNT_INSTALMENT_FUTURE': ['min', 'max', 'std', 'median'],
                    'SK_DPD': ['max', 'mean'],
                    'SK_DPD_DEF': ['max', 'mean'],
                    'NAME_CONTRACT_STATUS_Active': ['mean','sum'],
                    'NAME_CONTRACT_STATUS_Completed': ['mean','sum'],
                    'NAME_CONTRACT_STATUS_Signed': ['mean','sum'],
                    'NAME_CONTRACT_STATUS_Rare': ['mean','sum']}

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # 1:kredi zamaninda kapanmamis 0:kredi zamaninda kapanmis
    # POS_CNT_INSTALMENT_FUTURE: ÖNCEKİ KREDİNİN ÖDENMESİ İÇİN KALAN TAKSİTLER
    # POS_NAME_CONTRACT_STATUS: AY BOYUNCA SÖZLEŞME DURUMU
    # ÖNCEKİ KREDİDE KALAN TAKSİT SIFIRA EŞİTSE VE TAMAMLANMIŞ KREDİSİ SIFIRA EŞİTSE KREDİ ZAMANINDA TAMAMLANMAMIŞ
    pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'] = (pos_agg['POS_CNT_INSTALMENT_FUTURE_MIN'] == 0) & (pos_agg['POS_NAME_CONTRACT_STATUS_Completed_SUM'] == 0)
    pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'] = [1 if i == True else 0 for i in pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME']]

    # # Count pos cash accounts
    pos_agg['POS_NEW_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    import pandas as pd
    import numpy as np
    ins = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv',nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category)
    ins['NEW_PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']  # Kredi ödeme yüzdesi
    ins['NEW_PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']  # Toplam kalan borç
    # Days past due and days before due (no negative values)
    ins['NEW_DPE'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT'] # Ödeme tarihinden ne kadar önce ya da sonra ödedi
    ins['NEW_DPE'] = ins['NEW_DPE'].map(lambda x: 1 if x < 0 else 0) # Ödeme tarihini geçti mi geçmedi mi (1: geç ödedi, 0: erken ödedi)

    # ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    # ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    # ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    # ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    aggregations = {'NUM_INSTALMENT_VERSION': ['nunique'],
                    'NUM_INSTALMENT_NUMBER': ['max', 'mean', 'sum', 'median', 'std'],
                    'DAYS_INSTALMENT': ['max', 'mean', 'sum', 'std'],
                    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
                    'AMT_INSTALMENT': ['min','max', 'mean', 'sum', 'std'],
                    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
                    'NEW_DPE': ['max', 'mean', 'sum', 'median', 'std'],
                    'NEW_PAYMENT_PERC': ['max', 'mean', 'sum', 'std', 'median'],
                    'NEW_PAYMENT_DIFF': ['max', 'mean', 'sum', 'std', 'median']}
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv', nrows= num_rows)
    cc["NAME_CONTRACT_STATUS"] = np.where(~(cc["NAME_CONTRACT_STATUS"].isin(['Active', 'Completed', 'Signed'])), "Rare",cc["NAME_CONTRACT_STATUS"])
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

    grp = cc.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index=str, columns={'SK_ID_PREV': 'NUMBER_OF_LOANS_PER_CUSTOMER'})
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')
    grp = cc.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index=str,columns={'CNT_INSTALMENT_MATURE_CUM': 'NUMBER_OF_INSTALMENTS'})
    grp1 = grp.groupby(by=['SK_ID_CURR'])['NUMBER_OF_INSTALMENTS'].sum().reset_index().rename(index=str, columns={'NUMBER_OF_INSTALMENTS': 'TOTAL_INSTALMENTS_OF_ALL_LOANS'})
    cc = cc.merge(grp1, on=['SK_ID_CURR'], how='left')
    cc['INSTALLMENTS_PER_LOAN'] = (cc['TOTAL_INSTALMENTS_OF_ALL_LOANS'] / cc['NUMBER_OF_LOANS_PER_CUSTOMER']).astype('uint32')

    def day_past_due(dpd):
        # Önceki ayda vadeyi geçen gün sayısının sıfıra eşit olmaması durumlarını toplar
        dpd_list = dpd.tolist()
        days = 0
        for i in dpd_list:
            if i != 0:
                days = days + 1
        # Toplamda vadenin kaç kere geciktirildiğini ifade eder
        return days

    grp = cc.groupby(by=['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: day_past_due(x.SK_DPD)).reset_index().rename(index=str, columns={0: 'NUMBER_OF_DPD'})
    grp1 = grp.groupby(by=['SK_ID_CURR'])['NUMBER_OF_DPD'].mean().reset_index().rename(index=str, columns={'NUMBER_OF_DPD': 'DPD_COUNT'})

    cc = cc.merge(grp1, on=['SK_ID_CURR'], how='left')

    def min_rate(min_pay, total_pay):
        # minimum ödenmesi gereken miktardan daha az ödenmiş olan ayların yüzdeliğini hesaplar
        minimum = min_pay.tolist()
        total = total_pay.tolist()
        transactions = 0
        for i in range(len(minimum)):
            if total[i] < minimum[i]:
                transactions = transactions + 1

        return (transactions * 100) / len(minimum)

    grp = cc.groupby(by=['SK_ID_CURR']).apply(lambda x: min_rate(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index=str, columns={0: 'PERCENTAGE_MIN_MISSED_PAYMENTS'})
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')
    cc.head()

    #############################

    grp = cc.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index=str, columns={
        'AMT_DRAWINGS_ATM_CURRENT': 'DRAWINGS_ATM'})  # Önceki kredide ATM'den çekilen tutar
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    grp = cc.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={
        'AMT_DRAWINGS_CURRENT': 'DRAWINGS_TOTAL'})  # Önceki kredide çekilen tutar
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    cc['CASH_CARD_RATIO1'] = (cc['DRAWINGS_ATM'] / cc['DRAWINGS_TOTAL']) * 100  # ATM den cektigi nakit / toplam cektigi
    del cc['DRAWINGS_ATM']
    del cc['DRAWINGS_TOTAL']

    grp = cc.groupby(by=['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index=str, columns={'CASH_CARD_RATIO1': 'CASH_CARD_RATIO'})
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    grp = cc.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'TOTAL_DRAWINGS'})  # Önceki kredinin olduğu ay boyunca çekilen miktar (toplamı)
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    grp = cc.groupby(by=['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={'CNT_DRAWINGS_CURRENT': 'NUMBER_OF_DRAWINGS'})  # Önceki kredide bu aydaki çekimlerin sayısı (toplamı)
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    cc['DRAWINGS_RATIO1'] = (cc['TOTAL_DRAWINGS'] / cc['NUMBER_OF_DRAWINGS']) * 100  # yüzdelik olarak ifade edilmiyor, genişletme yapıldı
    del cc['TOTAL_DRAWINGS']
    del cc['NUMBER_OF_DRAWINGS']

    grp = cc.groupby(by=['SK_ID_CURR'])['DRAWINGS_RATIO1'].mean().reset_index().rename(index=str, columns={'DRAWINGS_RATIO1': 'DRAWINGS_RATIO'})
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    del cc['DRAWINGS_RATIO1']

    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ["sum", "mean"],
        'AMT_BALANCE': ["sum", "mean", "min", "max", 'std'],
        'AMT_CREDIT_LIMIT_ACTUAL': ["sum", "mean"],

        'AMT_DRAWINGS_ATM_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_DRAWINGS_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_DRAWINGS_OTHER_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_DRAWINGS_POS_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_INST_MIN_REGULARITY': ["sum", "mean", "min", "max", 'std'],
        'AMT_PAYMENT_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_PAYMENT_TOTAL_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_RECEIVABLE_PRINCIPAL': ["sum", "mean", "min", "max", 'std'],
        'AMT_RECIVABLE': ["sum", "mean", "min", "max", 'std'],
        'AMT_TOTAL_RECEIVABLE': ["sum", "mean", "min", "max", 'std'],

        'CNT_DRAWINGS_ATM_CURRENT': ["sum", "mean"],
        'CNT_DRAWINGS_CURRENT': ["sum", "mean", "max"],
        'CNT_DRAWINGS_OTHER_CURRENT': ["mean", "max"],
        'CNT_DRAWINGS_POS_CURRENT': ["sum", "mean", "max"],
        'CNT_INSTALMENT_MATURE_CUM': ["sum", "mean", "max", "min", 'std'],
        'SK_DPD': ["sum", "mean", "max"],
        'SK_DPD_DEF': ["sum", "mean", "max"],

        'NAME_CONTRACT_STATUS_Active': ["sum", "mean", "min", "max", 'std'],
        'INSTALLMENTS_PER_LOAN': ["sum", "mean", "min", "max", 'std'],

        'DPD_COUNT': ["mean"],
        'PERCENTAGE_MIN_MISSED_PAYMENTS': ["mean"],
        'CASH_CARD_RATIO': ["mean"],
        'DRAWINGS_RATIO': ["mean"]})

    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    return cc_agg


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    import re

    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :150].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(12, 28))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug=False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        br = bureau_and_balance(num_rows)
        print("Bureau df shape:", br.shape)
        df = df.join(br, how='left', on='SK_ID_CURR')
        del br
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds=10, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()
