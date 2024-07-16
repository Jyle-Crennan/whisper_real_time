import csv
import nltk.tokenize as tk
import en_core_web_sm

from nltk.stem import WordNetLemmatizer
from string import punctuation


def get_stopwords(stopword_file) -> list:
    stopwords = []
    try:
        # Collect all stopwords from asl_stopwords.csv; add to it at any time
        with open(stopword_file, newline="") as sw_file:
            for word in csv.reader(sw_file):
                stopwords.append(word[0])
    except FileNotFoundError:
        pass
    return stopwords


def get_transcription(transcription_file) -> str:
    transcription = ''
    try:
        # Convert transcription text file to a string
        with open(transcription_file, "r") as txt_obj:
            transcription = txt_obj.read()
            txt_obj.close()
    except FileNotFoundError:
        pass
    return transcription


def get_proper_nouns(transcription) -> list:
    # Proper noun extraction
    proper_nouns = []
    nlp = en_core_web_sm.load()
    for word in nlp(transcription):
        if word.pos_ == 'PROPN':
            proper_nouns.append(word.text)
    return proper_nouns


def get_sentences(transcription) -> list:
    return tk.sent_tokenize(transcription)


def tokenize_sentences(sentences) -> list[list]:
    return [tk.word_tokenize(sentence) for sentence in sentences]


def remove_preliminaries(words, *prelims) -> list:
    # Remove all punctuation/stopwords from token list; leaves only words
    tokenized_words = []
    for i in range(len(words)):
        for j in range(len(words[i])):
            token = words[i][j].lower()
            for prelim in prelims:
                if token not in punctuation and token not in prelim:
                    tokenized_words[i].append(token)
    return tokenized_words


def main():
    stopwords = get_stopwords("asl_stopwords.csv")
    transcription = get_transcription("transcription.txt")
    proper_nouns = get_proper_nouns(transcription)
    sentences = get_sentences(transcription)
    words = tokenize_sentences(sentences)
    tokenized_words = remove_preliminaries(words, stopwords)


if __name__ == "__main__":
    main()
