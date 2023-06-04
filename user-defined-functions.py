def remove_stopwords(text):
    """
    Remove stopwords and lowerize words in a text.

    Parameters
    ----------
    text : str
        Text to remove stopwords from.
    
    Returns
    -------
    str

    Notes
    -----
    This function can be used with the pd.Series.apply() method.
    Example usage: df['Clean Data'] = df['Text'].apply(remove_stopwords)
    """
    tokens = nltk.word_tokenize(text)
    filtered = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered)