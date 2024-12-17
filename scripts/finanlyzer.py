import pandas as pd
import os
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Function to convert date formats
def convert_date_format(date_str):
    if '-' in date_str:  # yyyy-mm-dd format
        date = pd.to_datetime(date_str, errors='coerce')
        return date.strftime('%m/%d/%Y') if not pd.isnull(date) else np.nan
    elif '/' in date_str:  # mm/dd/yyyy format
        date = pd.to_datetime(date_str, format='%m/%d/%Y %H:%M', errors='coerce')
        return date.strftime('%m/%d/%Y') if not pd.isnull(date) else np.nan
    else:
        return np.nan

# Function to align and save closing prices
def align_and_save_closing_prices(filtered_data, historical_paths, output_dir):
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    os.makedirs(output_dir, exist_ok=True)

    def align_data(stock_symbol, filtered_data, stock_data):
        filtered_stock = filtered_data[filtered_data['stock'] == stock_symbol]
        merged_stock = pd.merge(filtered_stock, stock_data[['Date', 'Close']], left_on='Date', right_on='Date', how='inner')
        merged_stock['stock'] = stock_symbol
        return merged_stock[['Date', 'headline', 'url', 'publisher', 'Close', 'stock']]

    for symbol, path in historical_paths.items():
        stock_data = pd.read_csv(path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        aligned_data = align_data(symbol, filtered_data, stock_data)
        output_file = os.path.join(output_dir, f'final_aligned_{symbol.lower()}_closing_data.csv')
        aligned_data.to_csv(output_file, index=False)

# Sentiment analysis functions
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

def categorize_sentiment(score, positive_threshold=0.05, negative_threshold=-0.05):
    if score >= positive_threshold:
        return 'Positive'
    elif score <= negative_threshold:
        return 'Negative'
    else:
        return 'Neutral'

def align_and_save_closing_prices_with_sentiment(filtered_data, historical_paths, output_dir):
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()

    for i, (symbol, path) in enumerate(historical_paths.items()):
        stock_data = pd.read_csv(path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        filtered_stock = filtered_data[filtered_data['stock'] == symbol]
        merged_stock = pd.merge(filtered_stock, stock_data[['Date', 'Close']], left_on='Date', right_on='Date', how='inner')
        merged_stock['stock'] = symbol
        
        merged_stock['sentiment'] = merged_stock['headline'].apply(analyze_sentiment)
        merged_stock['sentiment_category'] = merged_stock['sentiment'].apply(categorize_sentiment)

        output_file = os.path.join(output_dir, f'final_aligned_{symbol.lower()}_closing_data_with_sentiment.csv')
        merged_stock.to_csv(output_file, index=False)

        sns.histplot(merged_stock['sentiment'], bins=30, kde=True, color='skyblue', ax=axes[i])
        axes[i].axvline(x=-0.05, color='red', linestyle='--', label='Negative Threshold')
        axes[i].axvline(x=0.05, color='green', linestyle='--', label='Positive Threshold')
        axes[i].axvline(x=0, color='gray', linestyle='--', label='Neutral Threshold')
        axes[i].set_title(f'Distribution of Sentiment Scores for {symbol}')
        axes[i].set_xlabel('Sentiment Score')
        axes[i].set_ylabel('Frequency')
        axes[i].legend(title='Sentiment Thresholds')

        sentiment_counts = merged_stock['sentiment_category'].value_counts()
        print(f'Sentiment counts for {symbol}:')
        print(sentiment_counts)
        print("\n")

    if len(historical_paths) < 6:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()

# Function to process data
def process_data(data, filter_dates=None):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    if filter_dates:
        start_date, end_date = filter_dates
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    df['Return'] = df['Close'].pct_change()
    df.dropna(subset=['Return', 'sentiment'], inplace=True)
    
    return df

# Function to get the date range from data
def get_date_range(data):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    return start_date, end_date

# Function to interpret correlation coefficients
def get_correlation_interpretation(correlation):
    if correlation == 1.0:
        return "Perfect positive association"
    elif 0.8 <= correlation < 1.0:
        return "Very strong positive association"
    elif 0.6 <= correlation < 0.8:
        return "Strong positive association"
    elif 0.4 <= correlation < 0.6:
        return "Moderate positive association"
    elif 0.2 <= correlation < 0.4:
        return "Weak positive association"
    elif 0.0 <= correlation < 0.2:
        return "Very weak positive or no association"
    elif -0.2 < correlation <= 0.0:
        return "Very weak negative or no association"
    elif -0.4 < correlation <= -0.2:
        return "Weak negative association"
    elif -0.6 < correlation <= -0.4:
        return "Moderate negative association"
    elif -0.8 < correlation <= -0.6:
        return "Strong negative association"
    elif correlation <= -0.8:
        return "Very strong negative association"
    else:
        return "Invalid correlation coefficient"

# Function to plot and calculate correlations
def plot_and_calculate_correlations(data_paths, filter_dates=None):
    processed_data = {}
    for symbol, path in data_paths.items():
        df = pd.read_csv(path)
        
        # Get full date range
        start_date_full, end_date_full = get_date_range(df)
        
        if filter_dates:
            start_date_filtered, end_date_filtered = filter_dates
            df_filtered = process_data(df, filter_dates)
            processed_data[symbol] = df_filtered
            print(f'{symbol} - Filtered Date Range: {start_date_filtered} to {end_date_filtered}')
        else:
            processed_data[symbol] = process_data(df, (start_date_full, end_date_full))
            
        print(f'{symbol} - Full Date Range: {start_date_full} to {end_date_full}')
        print('---')
        
    colors = {
        'AAPL': 'blue',
        'AMZN': 'green',
        'GOOG': 'red',
        'NVDA': 'purple',
        'TSLA': 'orange'
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()

    correlations = {}
    for i, (symbol, df) in enumerate(processed_data.items()):
        correlation = df['Return'].corr(df['sentiment'])
        correlations[symbol] = correlation

        sns.scatterplot(x='sentiment', y='Return', data=df, ax=axes[i], color=colors[symbol])
        axes[i].set_title(f'{symbol}: Sentiment vs. Return\nCorrelation: {correlation:.2f}')
        axes[i].set_xlabel('Sentiment Score')
        axes[i].set_ylabel('Daily Return')
        
    if len(processed_data) < 6:
        fig.delaxes(axes[len(processed_data)])

    plt.tight_layout()
    plt.show()

    print("Pearson Correlations between Daily Returns and Sentiment Scores:")
    for symbol, corr in correlations.items():
        interpretation = get_correlation_interpretation(corr)
        print(f"{symbol}: {corr:.2f} - {interpretation}")


# Aggregate Average Sentiment

def compute_and_merge_sentiment(df, date_col='Date', sentiment_col='sentiment'):
    df[date_col] = pd.to_datetime(df[date_col])
    avg_sentiment_df = df.groupby(date_col)[sentiment_col].mean().reset_index()
    avg_sentiment_df.rename(columns={sentiment_col: 'Average_Daily_Sentiment'}, inplace=True)
    merged_df = pd.merge(df, avg_sentiment_df, on=date_col, how='left')
    
    return merged_df


# Function to process average sentiment data
def process_data_avg(data_paths, avg_sentiment_paths, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    for symbol, data_path in data_paths.items():
        df = pd.read_csv(data_path)
        avg_sentiment_df = pd.read_csv(avg_sentiment_paths[symbol])
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        avg_sentiment_df['Date'] = pd.to_datetime(avg_sentiment_df['Date'], errors='coerce')
        
        combined_df = pd.merge(df, avg_sentiment_df[['Date', 'Average_Daily_Sentiment']], on='Date', how='inner')
        combined_df['Return'] = combined_df['Close'].pct_change()
        combined_df.dropna(subset=['Return', 'Average_Daily_Sentiment'], inplace=True)
        
        output_file = os.path.join(output_directory, f'processed_{symbol.lower()}_data_with_avg_sentiment.csv')
        combined_df.to_csv(output_file, index=False)
        print(f'Processed data for {symbol} saved to {output_file}')

# Function to plot and calculate correlations with average sentiment
# Function to plot and calculate correlations
def plot_and_calculate_correlations_avg(data_paths, filter_dates=None):
    processed_data_avg = {}
    for symbol, path in data_paths.items():
        df = pd.read_csv(path)
        
        # Get full date range
        start_date_full, end_date_full = get_date_range(df)
        
        if filter_dates:
            start_date_filtered, end_date_filtered = filter_dates
            df_filtered = process_data(df, filter_dates)
            processed_data_avg[symbol] = df_filtered
            print(f'{symbol} - Filtered Date Range: {start_date_filtered} to {end_date_filtered}')
        else:
            processed_data_avg[symbol] = process_data(df, (start_date_full, end_date_full))
            
        print(f'{symbol} - Full Date Range: {start_date_full} to {end_date_full}')
        print('---')
        
    colors = {
        'AAPL': 'blue',
        'AMZN': 'green',
        'GOOG': 'red',
        'NVDA': 'purple',
        'TSLA': 'orange'
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()

    correlations = {}
    for i, (symbol, df) in enumerate(processed_data_avg.items()):
        correlation = df['Return'].corr(df['Average_Daily_Sentiment'])
        correlations[symbol] = correlation

        sns.scatterplot(x='Average_Daily_Sentiment', y='Return', data=df, ax=axes[i], color=colors[symbol])
        axes[i].set_title(f'{symbol}: Average_Daily_Sentiment vs. Return\nCorrelation: {correlation:.2f}')
        axes[i].set_xlabel('Sentiment Score')
        axes[i].set_ylabel('Daily Return')
        
    if len(processed_data_avg) < 6:
        fig.delaxes(axes[len(processed_data_avg)])

    plt.tight_layout()
    plt.show()

    print("Pearson Correlations between Daily Returns and Sentiment Scores:")
    for symbol, corr in correlations.items():
        interpretation = get_correlation_interpretation(corr)
        print(f"{symbol}: {corr:.2f} - {interpretation}")