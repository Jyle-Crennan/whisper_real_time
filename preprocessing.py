import csv
import nltk.tokenize as tk
import en_core_web_sm
import contractions
import spacy

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
    # Split transcription into sentences based off punctuation
    return tk.sent_tokenize(transcription)


def tokenize_sentences(sentences) -> list[list]:
    # Break apart sentences into smaller components; includes partial contractions and punctuation
    return [tk.word_tokenize(sentence) for sentence in sentences]


def remove_preliminaries(tokens, *prelims) -> list:
    # Remove all punctuation/stopwords from token list; leaves only words
    words = [[] for l in range(len(tokens))]
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            token = tokens[i][j].lower()
            for prelim in prelims:
                if token not in punctuation and token not in prelim:
                    words[i].append(token)
    return words


def expand_contractions(transcription) -> str:
    # Deals with contracted words, so they do not have to be removed after tokenization
    return contractions.fix(transcription)


def spacify(transcription) -> list:
    features = []
    nlp = spacy.load('en_core_web_sm')
    text = nlp(transcription)
    for token in text:
        token_feats = {
            'token': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'shape': token.shape_,
            'is_aplha': token.is_alpha,
            'is_stop': token.is_stop
        }
        #features.append([feature for feature in token_feats.values()])
        features.append(token_feats)
    return features


def main():
    transcription = expand_contractions(get_transcription("transcription.txt"))
    #print(transcription)

    stopwords = get_stopwords("asl_stopwords.csv")
    #print(stopwords)
    proper_nouns = get_proper_nouns(transcription)
    #print(proper_nouns)

    sentences = get_sentences(transcription)
    #print(sentences)
    tokens = tokenize_sentences(sentences)
    #print(tokens)
    words = remove_preliminaries(tokens, stopwords)
    #print(words)

    for feature in spacify(transcription):
        print([feat for feat in feature.values()])


if __name__ == "__main__":
    main()
