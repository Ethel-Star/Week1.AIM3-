# Notebooks Description

# A Data-Driven Exploration of Financial News and Stock Market Dynamics
Project Overview
This repository contains Jupyter notebooks for analyzing financial news data to enhance predictive analytics at Nova Financial Solutions. The project focuses on two main tasks:

# Sentiment Analysis: Analyzing the tone and sentiment of financial news headlines.
Correlation Analysis: Investigating the relationship between sentiment scores and stock price movements.
Notebooks
1. EDA_Financial_Headlines.ipynb
Purpose: Perform sentiment analysis on financial news headlines.
Key Steps:
Load and preprocess the financial news dataset.
Extract sentiment scores using NLP techniques.
Visualize sentiment distribution and trends.

3. TALib_Quant_Analysis.ipynb
Purpose: Assess stock performance using technical indicators from TA-Lib.
Key Steps:
Load historical stock price data.
Apply TA-Lib technical indicators (e.g., moving averages, RSI, Bollinger Bands).
Analyze and visualize stock performance and trends based on these indicators.
2. Correlation_Analysis.ipynb
Purpose: Analyze the correlation between sentiment scores and stock price changes.
Key Steps:
Merge sentiment data with stock price data.
Track stock price changes around publication dates of news articles.
Compute and visualize correlations between sentiment and stock performance.

pip install pandas numpy matplotlib seaborn textblob yfinance nltk scikit-learn statsmodels TA-Lib plotly pypfopt

# Usage

Run the Notebooks:

Open each notebook (EDA_Financial_Headlines.ipynb and TALib_Quant_Analysis.ipynb) in Jupyter Notebook or JupyterLab.
Follow the instructions and code cells to perform the analyses.
Review Results:

Analyze the outputs and visualizations generated in the notebooks.
Modify and experiment with the code to suit specific analysis needs.
