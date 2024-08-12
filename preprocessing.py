import csv
import string
import nltk.tokenize as tk
import contractions
import spacy
import spacy_experimental

# The other file
import transcribe_demo as td

from string import punctuation
from spacy.tokens.span_group import SpanGroup
from spacy.tokens import Span

# spaCy models
import en_core_web_sm
import en_coreference_web_trf

# Stopwords are the words that are NOT to be used in the finalized ASL sentence
# Mostly articles and conjunctions
# NOT finalized, add as you go
stopwords = ('a', 'an', 'the', 'of', 'to', 'be', 'are', 'is', 'as', 'so', 'if', 'it', 'for')


# Get transcription from text file, if needed
# (Previously used in testing)
def get_transcription(transcription_file) -> str:
    transcription = ''
    try:
        with open(transcription_file, "r") as txt_obj:
            transcription = txt_obj.read()
            txt_obj.close()
    except FileNotFoundError:
        pass
    return transcription


# Proper noun extraction
def get_proper_nouns(doc) -> list:
    return [word.text for word in doc if word.pos_ == 'PROPN']


def get_corefs(doc, include_heads) -> list[list[str]]:
    # Convert the spaCy span object to a dict; extracts coref clusters from values
    # Ex. [[Emma said, she thinks], [Nelson really, he goes]]
    coref_clusters = [[str(span) for span in list(coref)] for coref in dict(doc.spans).values()]
    # Only extract the heads from the clusters (front half of coref_clusters)
    # Ex. [[Emma, she], [Nelson, he]]
    head_clusters = coref_clusters[:len(coref_clusters)//2]
    if include_heads:
        return head_clusters
    else:
        return coref_clusters


# Split transcription into sentences based off punctuation
def get_sentences(transcription) -> list:
    return tk.sent_tokenize(transcription)


# Break apart sentences into smaller components; includes partial contractions and punctuation
def tokenize_sentences(sentences) -> list[list]:
    return [tk.word_tokenize(sentence) for sentence in sentences]


# Remove all punctuation/stopwords from token list; leaves only words
def remove_preliminaries(tokens, *prelims) -> list:
    length: int = len(tokens)
    words = [[] for l in range(length)]
    for i in range(length):
        # Yes, these loops can be optimized
        # No, I do not care
        for j in range(len(tokens[i])):
            token: str = tokens[i][j]
            for prelim in prelims:
                if token.lower() not in prelim:
                    words[i].append(token)
    return words


# Deals with contracted words, so they do not have to be removed after tokenization
def expand_contractions(transcription) -> str:
    return contractions.fix(transcription)


# Collect a feature set for each word to make them easier to process in the future
# A 'token' is literally just the word/punctuation
# A 'lemma' is the base of a word, not necessarily the root word
# A 'pos' is just the part of speech (see spacy doc for list of abreviations)
def spacify(doc) -> list[dict]:
    features = []
    corefs = get_corefs(doc, True)
    for token in doc:
        token_feats = {
            'token': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_
        }
        for coref in corefs:
            if token.text in coref:
                token_feats['coref'] = [n for n in coref if n != token.text]
                break
            else:
                token_feats['coref'] = []
        features.append(token_feats)
    return features


# The purpose of this function is to use basic rules of ASL glossing to convert the preprocessed sentences into an ASL glossed sentence
# Need to create a glossary of tagged text for training (absolutely NOT doing that by hand alone)
# Currently stuck here
def aslify(features: list) -> str:
    ...


def main():

    # Load the pre-trained spaCy models
    nlp = en_core_web_sm.load()
    nlp_coref = en_coreference_web_trf.load()
    nlp_coref.replace_listeners('transformer', 'coref', ['model.tok2vec'])
    nlp_coref.replace_listeners('transformer', 'span_resolver', ['model.tok2vec'])
    nlp.add_pipe('coref', source=nlp_coref)
    nlp.add_pipe('span_resolver', source=nlp_coref)

    # Run the transcribe_demo file, and get the transcript as a str
    td.main()
    transcription = expand_contractions(td.get_transcript(td.transcription))
    print(transcription)
    doc = nlp(transcription)

    proper_nouns = get_proper_nouns(doc)

    sentences = get_sentences(transcription)

    tokens = tokenize_sentences(sentences)
    words = remove_preliminaries(tokens, stopwords, punctuation)

    for test in spacify(doc):
        print(test)


if __name__ == "__main__":
    main()
