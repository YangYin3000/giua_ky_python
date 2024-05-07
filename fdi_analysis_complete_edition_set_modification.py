import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
#doc file
df_fdi_cPartners = pd.read_csv('fdi_country_partners_vi .csv',encoding='utf-8')
df_fdi_industry = pd.read_csv('fdi_industry_vi.csv',encoding='utf-8')
df_gnp = pd.read_csv('Vietnam-gnp-gross-national-product.csv',encoding='utf-8')
#khai báo mạng năm
year = np.array([2015,2016,2017,2018,2019,2020,2021,2022])
#làm sạch dữ liệu của dataframe GNP
df_gnp['year'] = pd.to_numeric(df_gnp['year'])
df_gnp[' GNP (Billions of USD)'] = pd.to_numeric(df_gnp[' GNP (Billions of USD)'])

#clean data
def clean_dataframe(df):
    df_clean = df.copy()
    df_clean.fillna(value=0, inplace=True)
    df_clean.replace(' -   ', 0, inplace=True)
    df_clean.replace(',', '', inplace=True)
    df_clean = df_clean.astype('str')
    for column in df_clean.columns[2:]:
        df_clean[column]=df_clean[column].apply(lambda x: str(x.replace(',', '').replace('(', '-').replace(')', '')) if isinstance(x, str) else x)
    #đổi thành dữ liệu kiểu số để tính toán
    df_clean[' Vốn đăng ký cấp mới (triệu USD) '] = pd.to_numeric(df_clean[' Vốn đăng ký cấp mới (triệu USD) '])
    df_clean[' Vốn đăng ký điều chỉnh (triệu USD) '] = pd.to_numeric(df_clean[' Vốn đăng ký điều chỉnh (triệu USD) '])
    df_clean[' Giá trị góp vốn, mua cổ phần (triệu USD) '] = pd.to_numeric(df_clean[' Giá trị góp vốn, mua cổ phần (triệu USD) '])
    df_clean[' Tổng FDI '] = df_clean[' Vốn đăng ký cấp mới (triệu USD) '] + df_clean[' Vốn đăng ký điều chỉnh (triệu USD) '] + df_clean[' Giá trị góp vốn, mua cổ phần (triệu USD) ']

    return df_clean

df_countryPartners = clean_dataframe(df_fdi_cPartners)
df_industry = clean_dataframe(df_fdi_industry)

# # Tính tổng FDI qua các năm
def sum_fdi(df):
    return df.groupby('Năm')[' Tổng FDI '].sum()

# # Phân tích mô tả
# Tìm hiểu cơ bản các xu hướng cơ bản về FDI qua các năm
def descriptive_analysis(df_countryPartners, year):
    plt.figure(figsize=(12,6))
    plt.bar(year, sum_fdi(df_countryPartners).values)
    plt.title('XU HƯỚNG FDI QUA CÁC NĂM')
    plt.xlabel('Năm')
    plt.ylabel('Số tiền đầu tư (tính bằng triệu USD)')
    plt.savefig('pic_1.png')
    plt.show()

# # Phân tích xu hướng
# Xác định các xu hướng và thay đổi trong dòng vốn FDI
def trend_analysis(df_countryPartners,year):
    reroll_FDI = df_countryPartners.groupby('Năm')[' Vốn đăng ký điều chỉnh (triệu USD) '].sum()
    print(reroll_FDI.to_string())
    buy_FDI = df_countryPartners.groupby('Năm')[' Giá trị góp vốn, mua cổ phần (triệu USD) '].sum()
    bar_width = 0.2
#vẽ đồ thị
    plt.figure(figsize=(12,6))#chỉnh kích thước của khung biểu đồ
    plt.bar(year-bar_width,sum_fdi(df_countryPartners),color = 'blue',width= 0.2)
    plt.bar(year,reroll_FDI,color = 'red',width= 0.2)
    plt.bar(year+bar_width,buy_FDI,color = 'black',width= 0.2)
    plt.title('XU HƯỚNG FDI VÀ SỰ THAY ĐỔI DÒNG VỐN QUA CÁC NĂM')
    plt.legend(['FDI tổng qua các năm','Vốn đăng kí điều chỉnh','Giá trị góp vốn, mua cổ phần'])
    plt.xlabel('Năm')
    plt.ylabel('Số tiền đầu tư (tính bằng triệu USD)')
    plt.savefig('pic_2.png')
    plt.show()

# # Phân tích nguồn
# Phân tích quốc gia nào là nhà đầu tư lớn tại Việt Nam
def soure_analysis(df_countryPartners):
    df_countryPartners['Đối tác'] = df_countryPartners['Đối tác'].str.strip()
    source_analysis = df_countryPartners.groupby('Đối tác')[' Tổng FDI '].sum().sort_values(ascending=False)

    plt.figure(figsize=(12,6))
    source_analysis.head(10).plot(kind='bar', title='Top 10 Quốc Gia Đầu Tư Lớn Tại Việt Nam')
    plt.ylabel('Triệu USD')
    plt.savefig('pic_3.png')
    plt.show()


# # # Phân tích ngành
# Xác định ngành nào thu hút FDI nhiều nhất
def sector_analysis(df):
# tính tổng FDI và sắp xếp theo ngành
    sector_analysis = df.groupby('Ngành')[' Tổng FDI '].sum().sort_values(ascending=False)
    print(sector_analysis.head(10))
#vẽ đồ thị
    plt.figure(figsize=(12,6))
    sector_analysis.head(10).plot(kind='bar', title='Top 10 ngành thu hút FDI nhiều nhất')
    plt.ylabel('Triệu USD')
    plt.savefig('pic_4.png')
    plt.show()

# # # Phân tích tác động
# # Cho dữ liệu GNP của việt nam từ năm 2015-2022 từ macrotrends.net
def analyze_enconomic_impact(df_gnp,year):
    plt.figure(figsize=(12,6))
    plt.plot(year, sum_fdi(df_countryPartners), marker='o')
    plt.plot(year, df_gnp[' GNP (Billions of USD)']*1000, marker='*')
    plt.title('ẢNH HƯỞNG CỦA FDI ĐẾN NỀN KINH TẾ')
    plt.legend(['FDI Tổng qua các năm','GNP qua các năm'])
    plt.xlabel('Năm')
    plt.ylabel('triệu USD')
    plt.grid(True) #dựng dạng lưới cho đồ thị
    plt.savefig('pic_5.png')
    plt.show()

# Dự báo xu hướng FDI và vẽ biểu đồ
def forecast_fdi_trend(df):
    # Chuyển đổi dữ liệu thành chuỗi thời gian
    ts = df.groupby('Năm')[' Tổng FDI '].sum()
    model = ARIMA(ts, order=(1, 1, 1))
    results = model.fit()
    plt.plot(ts, label='Thực Tế')
    plt.plot(results.fittedvalues, color='red', label='Dự Báo')
    plt.legend()
    plt.title('Dự Báo FDI')
    plt.ylabel('Triệu USD')
    plt.savefig('pic_6.png')
    plt.show()

descriptive_analysis(df_countryPartners, year)
trend_analysis(df_countryPartners,year)
soure_analysis(df_countryPartners)
sector_analysis(df_industry)
analyze_enconomic_impact(df_gnp,year)
forecast_fdi_trend(df_countryPartners)