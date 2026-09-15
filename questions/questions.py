import nltk
import sys
import math
import os
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files= {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                files[filename] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stopwords = set(nltk.corpus.stopwords.words("english"))
    punctuation = set(string.punctuation)
    words = nltk.word_tokenize(document.lower())
    return [word for word in words if word not in stopwords and word not in punctuation]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_documents = len(documents)
    word_doc_counts = {}
    for doc_words in documents.values():
        for word in doc_words:
            word_doc_counts[word] = word_doc_counts.get(word, 0) + 1

    idfs = {}
    for word, count in word_doc_counts.items():
        idfs[word] = math.log(num_documents / count)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = {}
    for filename, words in files.items():
        total_tf_idf = 0
        for word in query:
            tf = words.count(word)
            total_tf_idf += tf * idfs.get(word, 0)
        tf_idfs[filename] = total_tf_idf

    # Sort files by tf-idf score in descending order and return the top n filenames
    sorted_files = sorted(tf_idfs.keys(), key=lambda f: tf_idfs[f], reverse=True)
    return sorted_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = {}
    for sentence, words in sentences.items():
        score = sum(idfs.get(word, 0) for word in words if word in query)
        query_term_density = sum(1 for word in words if word in query) / len(words) if words else 0
        sentence_scores[sentence] = (score, query_term_density)

    # Sort sentences by score and query term density in descending order and return the top n sentences
    sorted_sentences = sorted(sentence_scores.keys(), key=lambda s: sentence_scores[s], reverse=True)
    return sorted_sentences[:n]


if __name__ == "__main__":
    main()
