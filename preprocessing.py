import csv
import nltk.tokenize as tk
import contractions
import spacy
import spacy_experimental

from string import punctuation
from spacy.tokens.span_group import SpanGroup
from spacy.tokens import Span

# spaCy models
import en_core_web_sm
import en_coreference_web_trf


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


def get_corefs(transcription):
    nlp = en_core_web_sm.load()
    nlp_coref = en_coreference_web_trf.load()
    nlp_coref.replace_listeners('transformer', 'coref', ['model.tok2vec'])
    nlp_coref.replace_listeners('transformer', 'span_resolver', ['model.tok2vec'])
    nlp.add_pipe('coref', source=nlp_coref)
    nlp.add_pipe('span_resolver', source=nlp_coref)
    doc = nlp(transcription)
    # Are you fucking shitting me?
    # An explicit cast from a non-iterable object type to a dict was the issue?
    corefs = dict(doc.spans)
    for coref in corefs.values():
        print(coref)


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
    # Returns a list of dicts
    features = []
    nlp = spacy.load('en_core_web_sm')
    text = nlp(transcription)
    for token in text:
        token_feats = {
            'token': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'coref': token
        }
        features.append(token_feats)
    return features


def aslify(features: list) -> str:
    return ''


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

    #for feature in spacify(transcription):
    #    print([feat for feat in feature.values()])

    get_corefs(transcription)


if __name__ == "__main__":
    main()
