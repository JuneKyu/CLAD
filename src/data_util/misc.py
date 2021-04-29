from torch.utils.data.dataset import Dataset
from torchnlp.encoders.encoder import Encoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import string
import re


def print_top_words(top_words_list, export_file, title=''):
    """
    Print the top words per context.
    :param top_words_list: list of lists of pairs (<word>, <count>) of top-k words in vocabulary by occurrence counts.
    :param export_file: path to export file.
    :param title: optional title.
    """

    n_contexts = len(top_words_list)

    export_txt = export_file + '.txt'
    txt_file = open(export_txt, 'a')

    if title:
        txt_file.write(f'{title}\n\n')

    for context in range(n_contexts):
        txt_file.write(f'Context {context:02}\n')

        for (word, count) in top_words_list[context]:
            txt_file.write(f'#{count:03}: {word}\n')

        txt_file.write(f'\n')

    txt_file.close()

    return


def get_correlation_matrix(a, eps=1e-08):
    """
    Compute correlation matrix (cosine similarity) of 2D-array a.
    The diagonal holds the norms of the row vectors (instead of cosine similarities of 1).
    """

    dot_products = np.dot(a, a.transpose())
    norms = np.sqrt(dot_products).diagonal()
    corr_mat = dot_products / (norms[:, np.newaxis] + eps)
    corr_mat = corr_mat / (norms[np.newaxis, :] + eps)
    corr_mat[np.diag_indices(len(norms))] = norms

    return corr_mat


def clean_text(text: str, rm_numbers=True, rm_punct=True, rm_stop_words=True, rm_short_words=True):
    """ Function to perform common NLP pre-processing tasks. """

    # make lowercase
    text = text.lower()

    # remove punctuation
    if rm_punct:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # remove numbers
    if rm_numbers:
        text = re.sub(r'\d+', '', text)

    # remove whitespaces
    text = text.strip()

    # remove stopwords
    if rm_stop_words:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text_list = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(text_list)

    # remove short words
    if rm_short_words:
        text_list = [w for w in text.split() if len(w) >= 3]
        text = ' '.join(text_list)

    return text
