import csv
import nltk.tokenize as tk
import en_core_web_sm

from nltk.stem import WordNetLemmatizer
from string import punctuation


def main():
    lemmatizer = WordNetLemmatizer()

    try:
        # Collect all stopwords from asl_stopwords.csv; add to it at any time
        with open("asl_stopwords.csv", newline="") as sw_file:
            stopwords = []
            for word in csv.reader(sw_file):
                stopwords.append(word[0])
        # Convert transcription text file to a string
        with open("transcription.txt", "r") as txt_obj:
            text = txt_obj.read()
            txt_obj.close()
    except FileNotFoundError:
        pass

    # Proper noun extraction
    nlp = en_core_web_sm.load()
    proper_nouns = []
    for word in nlp(text):
        if word.pos_ == 'PROPN':
            proper_nouns.append(word.text)
    print(proper_nouns)

    # Create tokenized sentences from transcription
    sentences = tk.sent_tokenize(text)
    words = [tk.word_tokenize(sentence) for sentence in sentences]
    tokenized_words = [[] for l in range(len(words))]

    # Remove all punctuation/stopwords from token list; leaves only words
    for i in range(len(words)):
        for j in range(len(words[i])):
            token = words[i][j].lower()
            if token not in punctuation and token not in stopwords:
                tokenized_words[i].append(token)


if __name__ == "__main__":
    main()
