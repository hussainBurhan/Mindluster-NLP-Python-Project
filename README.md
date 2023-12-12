# SMS Spam Detection and Text Analysis

## Project Overview

This project focuses on the analysis of SMS messages to detect spam using various text preprocessing and feature engineering techniques. 
Spam detection in SMS messages is a critical task in the realm of natural language processing and machine learning. 
The goal is to build a model that can accurately identify whether a given SMS message is spam or not.

## Learning Objectives

- Preprocess text data for machine learning tasks, including removing punctuation, tokenization, removing stopwords, stemming, and lemmatization.
- Use different techniques for text vectorization, including Count Vectorization, N-gram Vectorization, and Tf-Idf Vectorization.
- Perform feature engineering on text data, including extracting message length and punctuation percentage.
- Visualize and analyze the distribution of message length and punctuation percentage for spam and ham messages.

## Files

- `SMSSpamCollection`: The dataset containing SMS messages labeled as spam or ham.
- `script.py`: The Python script containing the code for text preprocessing, vectorization, feature engineering, and visualization.
- `README.md`: This file, providing an overview of the project.

## Project Execution

1. **Data Loading:** The SMS messages are loaded from the `SMSSpamCollection` file.
2. **Text Preprocessing:** The script employs various text preprocessing techniques, including removing punctuation, tokenization, removing stopwords, stemming, and lemmatization.
3. **Text Vectorization:** The text is vectorized using different techniques such as Count Vectorization, N-gram Vectorization, and Tf-Idf Vectorization.
4. **Feature Engineering:** Additional features such as message length and punctuation percentage are extracted to enhance the dataset.
5. **Visualization:** The distribution of message length and punctuation percentage is visualized for both spam and ham messages.

## Usage

1. Make sure you have Python and the required libraries (pandas, nltk, scikit-learn, matplotlib) installed.
2. Clone the repository to your local machine.
3. Run the `script.py` file to execute the text analysis and visualization.

## Dependencies

- pandas
- nltk
- scikit-learn
- matplotlib

## Output

The script outputs DataFrames showing the results of Count Vectorization, N-gram Vectorization, and Tf-Idf Vectorization. Additionally, visualizations of message length and punctuation percentage distributions are displayed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
