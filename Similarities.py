from jiwer import wer
import nltk 
import distance
import re, math
from collections import Counter
from nltk.corpus import stopwords
from nltk import tokenize
import argparse
import os
from pathlib import Path


# Global variables
WORD = re.compile(r'\w+')
sws = stopwords.words('portuguese')


# Stopwords removal
def text_normalized(text):
    palavras_tokenize = tokenize.word_tokenize(text, language='portuguese')
    filtered_sentence = list(filter(lambda x: x.lower() not in sws, palavras_tokenize))
    return " ".join(filtered_sentence)


# Cosine
def get_cosine_result(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def get_cosine(text1, text2):
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    cosine = get_cosine_result(vector1, vector2)
    return cosine


# Jaccard
def get_jaccard(text1, text2):
    jaccard = nltk.jaccard_distance(set(text1), set(text2))
    return jaccard


# Levenshtein
def get_levenshtein(text1, text2):
    levenshtein = distance.levenshtein(text1, text2)
    return levenshtein


# Word Error Rate
def get_wer(text1, text2):
    return wer(text1, text2)


# Punctuations
def get_pontuation(text):
    numberOfFullStops = 0
    numberOfQuestionMarks = 0
    numberOfExclamationMarks = 0
    numberOfCommaMarks = 0
    numberOfColonMarks = 0
    numberTotalPunctuation = 0

    for line in text:
        numberOfFullStops += line.count(".")
        numberOfQuestionMarks += line.count("?")
        numberOfExclamationMarks += line.count("!")
        numberOfCommaMarks += line.count(",")
        numberOfColonMarks += line.count(":")

    numberTotalPunctuation = numberOfFullStops + numberOfCommaMarks + numberOfQuestionMarks + numberOfExclamationMarks + numberOfColonMarks
    return numberOfFullStops, numberOfCommaMarks, numberOfQuestionMarks, numberOfExclamationMarks, numberOfColonMarks, numberTotalPunctuation


if __name__ == '__main__':

    lines = []
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, nargs=2, metavar='dir', help='directories of documents .txt. OBS.: First .txt might be the text recovered by API')
    args = parser.parse_args()
    filename = args.d
    for line in filename:
        lines.append(Path(line).read_text())

    # Documents or Texts
    test1 = re.sub(r'\n{1,}|\s{2,}', " ", lines[0])
    test2 = re.sub(r'\n{1,}|\s{2,}', " ", lines[1])

    print("BASE")
    print(test1)
    print("AMOSTRA")
    print(test2)

    # Get punctuation
    numberOfPunctuation = get_pontuation(test1)
    numberOfPunctuation2 = get_pontuation(test2)

    # Stopwords removal
    # test1 = text_normalized(test1)
    # test2 = text_normalized(test2)

    # Similatities results
    print("WER:", "%.2f" % get_wer(test1, test2))
    print("Jaccard:", "%.2f" % get_jaccard(test1, test2))
    print("Levenshtein:", get_levenshtein(test1, test2))
    print('Cosine:', "%.2f" % get_cosine(test1, test2))

    # Punctuation results
    print("PONTUAÇÃO BASE")
    print('Quantidade de Pontos:', numberOfPunctuation[0])
    print('Quantidade de Virgulas:', numberOfPunctuation[1])
    print('Quantidade de Interrogações:', numberOfPunctuation[2])
    print('Quantidade de Exclamações:', numberOfPunctuation[3])
    print('Quantidade de Dois Pontos:', numberOfPunctuation[4])
    print('Quantidade Total de Pontuações:', numberOfPunctuation[5])

    print("PONTUAÇÃO AMOSTRA")
    print('Quantidade de Pontos:', numberOfPunctuation2[0])
    print('Quantidade de Virgulas:', numberOfPunctuation2[1])
    print('Quantidade de Interrogações:', numberOfPunctuation2[2])
    print('Quantidade de Exclamações:', numberOfPunctuation2[3])
    print('Quantidade de Dois Pontos:', numberOfPunctuation2[4])
    print('Quantidade Total de Pontuações:', numberOfPunctuation2[5])
