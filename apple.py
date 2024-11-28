import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests




ticker_symbol = "AAPL"
apple_stock = yf.Ticker(ticker_symbol)


historical_data = apple_stock.history(start="2024-01-01")


shares = apple_stock.info.get("sharesOutstanding")
quarterly_net_income = apple_stock.quarterly_financials.loc['Net Income']

if shares is not None and not quarterly_net_income.isna().all():

    quarterly_dates = quarterly_net_income.index
    net_income_daily = pd.Series(
        data=quarterly_net_income.values,
        index=pd.to_datetime(quarterly_dates)
    ).resample('D').ffill()


    net_income_daily.index = net_income_daily.index.tz_localize(None)
    historical_data.index = historical_data.index.tz_localize(None)


    historical_data['Net Income'] = net_income_daily.reindex(historical_data.index, method='ffill')


    historical_data['EPS'] = historical_data['Net Income'] / shares


    correlation_matrix = historical_data[['Close', 'EPS']].corr()
    historical_data['P/E'] = historical_data['Close'] / historical_data['EPS']
    correlation_matrix_pe = historical_data[['Close', 'P/E']].corr()


    plt.figure(figsize=(10, 6))
    plt.plot(historical_data.index, historical_data['Close'], label='Stock Price (Close)')
    plt.title("Apple Stock Price (2024)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()


    company_info = {
        "Name": apple_stock.info.get("longName"),
        "Sector": apple_stock.info.get("sector"),
        "Industry": apple_stock.info.get("industry"),
        "Market Cap": apple_stock.info.get("marketCap"),
        "Shares Outstanding": shares
    }

    print("Company Information:")
    for key, value in company_info.items():
        print(f"{key}: {value}")


    print("\nFormula for EPS:")
    print("EPS = Net Income / Shares Outstanding")


    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Matrix between Close Price and EPS")
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(historical_data.index, historical_data['EPS'], label='EPS', color='orange')
    plt.plot(historical_data.index, historical_data['Close'], label='Stock Price (Close)', alpha=0.7)
    plt.title("EPS and Stock Price Correlation")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix_pe, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Matrix: Close Price and P/E")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(historical_data.index, historical_data['Close'], label="Stock Price", color="blue")
    plt.plot(historical_data.index, historical_data['P/E'], label="P/E Ratio", color="orange")
    plt.title("P/E and Stock Price Correlation")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()
#---------------------------------------------------------------------
    quarterly_financials = apple_stock.quarterly_financials.T


    historical_data = apple_stock.history(period="max")


    historical_data.index = historical_data.index.tz_localize(None)


    historical_data_quarterly = historical_data['Close'].resample('QE-DEC').mean()



    quarterly_financials = apple_stock.quarterly_financials.T


    quarterly_financials.index = quarterly_financials.index.tz_localize(None)


    quarterly_data = pd.DataFrame({
        "Close Price": historical_data_quarterly,
        "Revenue": quarterly_financials['Total Revenue'],
        "Net Income": quarterly_financials['Net Income']
    })


    quarterly_data.dropna(inplace=True)


    correlation_matrix_revenue = quarterly_data[['Close Price', 'Revenue']].corr()


    correlation_matrix_net_income = quarterly_data[['Close Price', 'Net Income']].corr()


    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix_revenue, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Matrix: Stock Price and Revenue")
    plt.show()


    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix_net_income, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Matrix: Stock Price and Net Income")
    plt.show()
#---------------------------------------
    api_key = '2fb1fdb943c1bf43f25336922632b230'
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={api_key}&file_type=json'


    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Ошибка при запросе данных CPI: {response.status_code}")
    data = response.json()


    cpi_data = pd.DataFrame(data['observations'])
    cpi_data['date'] = pd.to_datetime(cpi_data['date'])
    cpi_data.set_index('date', inplace=True)
    cpi_data['value'] = cpi_data['value'].astype(float)


    cpi_data = cpi_data.loc['2024-01-01':]


    cpi_data['Inflation Rate'] = cpi_data['value'].pct_change() * 100


    stock_data = yf.download('AAPL', start='2024-01-01')
    if stock_data.empty:
        raise ValueError("Данные о ценах акций не загружены. Проверьте символ акции или интернет-соединение.")


    stock_data['Stock Returns'] = stock_data['Adj Close'].pct_change() * 100


    cpi_monthly = cpi_data.resample('ME').last()
    stock_monthly = stock_data.resample('ME').last()


    fig, ax1 = plt.subplots(figsize=(12, 6))


    ax1.set_title("График инфляции и изменения цены акции (с 01.01.2024)", fontsize=14)
    ax1.plot(cpi_monthly.index, cpi_monthly['Inflation Rate'], color='blue', label='Inflation Rate')
    ax1.set_xlabel('Дата', fontsize=12)
    ax1.set_ylabel('Инфляция (%)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')


    ax2 = ax1.twinx()
    ax2.plot(stock_monthly.index, stock_monthly['Stock Returns'], color='green', label='Stock Returns')
    ax2.set_ylabel('Изменение цены акций (%)', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')


    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=10)

    plt.show()


#----------------------------------------------

    dollar_data = yf.download('DX-Y.NYB', start='2024-01-01')


    stock_data = yf.download('AAPL', start='2024-01-01')


    dollar_data['Dollar Change'] = dollar_data['Adj Close'].pct_change() * 100


    stock_data['Stock Returns'] = stock_data['Adj Close'].pct_change() * 100


    dollar_monthly = dollar_data.resample('ME').last()
    stock_monthly = stock_data.resample('ME').last()


    combined_data = pd.merge(dollar_monthly[['Dollar Change']], stock_monthly[['Stock Returns']], left_index=True,
                             right_index=True)
    combined_data.dropna(inplace=True)


    correlation_matrix = combined_data.corr()


    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Matrix: Dollar Change and Stock Returns")
    plt.show()


    fig, ax1 = plt.subplots(figsize=(12, 6))


    ax1.set_title("График колебания курса доллара и изменения цены акции (с 01.01.2024)", fontsize=14)
    ax1.plot(dollar_monthly.index, dollar_monthly['Dollar Change'], color='blue', label='Dollar Change')
    ax1.set_xlabel('Дата', fontsize=12)
    ax1.set_ylabel('Изменение курса доллара (%)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')


    ax2 = ax1.twinx()
    ax2.plot(stock_monthly.index, stock_monthly['Stock Returns'], color='green', label='Stock Returns')
    ax2.set_ylabel('Изменение цены акций (%)', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')


    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=10)

    plt.show()
#------------------------------------------------

    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-11-27"


    data = yf.download(ticker, start=start_date, end=end_date)


    if not data.empty:

        data_filtered = data[['Close', 'Volume']]


        correlation_matrix = data_filtered.corr()


        print("Матрица корреляции между ценой акций и объемом торгов:")
        print(correlation_matrix)


        plt.figure(figsize=(8, 6))
        plt.title('Матрица корреляции (Apple)')


        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Коэффициент корреляции')


        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=12)

        plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
        plt.yticks(ticks=range(len(correlation_matrix.index)), labels=correlation_matrix.index)
        plt.show()
    else:
        print("Не удалось загрузить данные для заданного периода.")
#-------------------------------------------------
#Объем торгов

    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-11-27"


    data = yf.download(ticker, start=start_date, end=end_date)


    if not data.empty:

        data_filtered = data[['Close', 'Volume']]


        correlation_matrix = data_filtered.corr()


        print("Матрица корреляции между ценой акций и объемом торгов:")
        print(correlation_matrix)


        plt.figure(figsize=(8, 6))
        plt.title('Матрица корреляции (Apple)')


        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Коэффициент корреляции')


        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=12)

        plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
        plt.yticks(ticks=range(len(correlation_matrix.index)), labels=correlation_matrix.index)
        plt.show()
    else:
        print("Не удалось загрузить данные для заданного периода.")
#--------------------------------
#Волатильность

    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-11-27"


    data = yf.download(ticker, start=start_date, end=end_date)


    if not data.empty:

        data['Volatility'] = data['High'] - data['Low']


        data_filtered = data[['Close', 'Volatility']]


        correlation_matrix = data_filtered.corr()


        print("Матрица корреляции между ценой акций и волатильностью:")
        print(correlation_matrix)


        plt.figure(figsize=(8, 6))
        plt.title('Матрица корреляции (Apple)')


        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Коэффициент корреляции')


        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=12)

        plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
        plt.yticks(ticks=range(len(correlation_matrix.index)), labels=correlation_matrix.index)
        plt.show()
    else:
        print("Не удалось загрузить данные для заданного периода.")
#----------------------------------------------------
#Дивиденды



    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-11-27"


    data = yf.download(ticker, start=start_date, end=end_date, actions=True)  # Включаем информацию о дивидендах


    if not data.empty:

        data_filtered = data[['Close', 'Dividends']].copy()


        data_filtered['Dividends'] = data_filtered['Dividends'].fillna(0)


        correlation_matrix = data_filtered.corr()


        print("Матрица корреляции между ценой акций и дивидендами:")
        print(correlation_matrix)


        plt.figure(figsize=(8, 6))
        plt.title('Матрица корреляции (Apple)')


        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Коэффициент корреляции')


        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=12)

        plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
        plt.yticks(ticks=range(len(correlation_matrix.index)), labels=correlation_matrix.index)
        plt.show()
    else:
        print("Не удалось загрузить данные для заданного периода.")
#-------------------------------------------------
#Скользящее среднее
    ticker = "AAPL"  #
    start_date = "2024-01-01"
    end_date = "2024-11-27"


    data = yf.download(ticker, start=start_date, end=end_date)


    if not data.empty:

        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_30'] = data['Close'].rolling(window=30).mean()


        data_filtered = data[['Close', 'SMA_10', 'SMA_30']].dropna()


        correlation_matrix = data_filtered.corr()


        print("Матрица корреляции между ценой акций и скользящими средними:")
        print(correlation_matrix)


        plt.figure(figsize=(8, 6))
        plt.title('Матрица корреляции (Apple)')


        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Коэффициент корреляции')


        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=12)

        plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
        plt.yticks(ticks=range(len(correlation_matrix.index)), labels=correlation_matrix.index)
        plt.show()
    else:
        print("Не удалось загрузить данные для заданного периода.")
#---------------------------------------------------------------------
#RSI

    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-11-27"


    data = yf.download(ticker, start=start_date, end=end_date)


    if not data.empty:

        data['Delta'] = data['Close'].diff()


        data['Gain'] = data['Delta'].where(data['Delta'] > 0, 0)
        data['Loss'] = -data['Delta'].where(data['Delta'] < 0, 0)


        data['Avg_Gain'] = data['Gain'].rolling(window=14).mean()
        data['Avg_Loss'] = data['Loss'].rolling(window=14).mean()


        data['RS'] = data['Avg_Gain'] / data['Avg_Loss']


        data['RSI'] = 100 - (100 / (1 + data['RS']))


        data_filtered = data[['Close', 'RSI']].dropna()


        correlation_matrix = data_filtered.corr()


        print("Матрица корреляции между ценой акций и RSI:")
        print(correlation_matrix)


        plt.figure(figsize=(8, 6))
        plt.title('Матрица корреляции (Apple)')


        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Коэффициент корреляции')


        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=12)

        plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
        plt.yticks(ticks=range(len(correlation_matrix.index)), labels=correlation_matrix.index)
        plt.show()
    else:
        print("Не удалось загрузить данные для заданного периода.")
#---------------------------------------------------------------------
#MACD

    ticker = "AAPL"  # Тикер для Apple
    start_date = "2024-01-01"
    end_date = "2024-11-27"


    data = yf.download(ticker, start=start_date, end=end_date)


    if not data.empty:

        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()


        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()


        data_filtered = data[['Close', 'MACD']].dropna()


        correlation_matrix = data_filtered.corr()


        print("Матрица корреляции между ценой акций и MACD:")
        print(correlation_matrix)


        plt.figure(figsize=(8, 6))
        plt.title('Матрица корреляции (Apple)')


        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Коэффициент корреляции')


        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=12)

        plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
        plt.yticks(ticks=range(len(correlation_matrix.index)), labels=correlation_matrix.index)
        plt.show()
    else:
        print("Не удалось загрузить данные для заданного периода.")
else:
    print("Error: Financial data is incomplete.")