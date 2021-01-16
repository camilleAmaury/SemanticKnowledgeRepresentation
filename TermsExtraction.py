import spacy
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopWords = set(STOP_WORDS)
stopWords.add("e.g")

def main():
    """
    document terms extraction
    """
    #inputs
    corpus_dir = r".\Corpus" #directory for corpus documents
    output_dir = r".\OutputDir" #result files output directory
    output_files = [output_dir+r"\ExtractedTerms0-205.txt", output_dir+r"\ExtractedTerms206-400.txt", output_dir+r"\ExtractedTerms401-600.txt"] #the path to save the extracted terms
    minFreq = 7 #minimum frequency threshold

    #compute tf for each term in the corpus
    tf= computerTf(corpus_dir)
    #if tf of the term is greater than minimum freq save it to the output file
    word_writed = 0
    
    file_output = 0
    terms_file = open(output_files[file_output], "w", errors='ignore')
    for term, score in tf.items():
        if score >= minFreq:
            word_writed += 1
            terms_file.write(str(term) + "\n")
            if word_writed == 205 or word_writed == 400:
                file_output+=1
                terms_file = open(output_files[file_output], "w", errors='ignore')
    print("\n > Word writed : {}".format(word_writed))

def removeArticles(text):
    #remove stop words from the begining of a NP
    words = text.split()
    if words[0] in stopWords:
        return text.replace(words[0]+ " ", "")
    return text

def computerTf(dir):
    alldocs = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    AllTerms = dict()
    for doc in alldocs:
        lines = open(doc, "r", errors='ignore').readlines()
        scale = 50
        scale_lines = [0] + list(range(0, len(lines), scale))
        if scale_lines[-1] != len(lines):
            scale_lines.append(len(lines))
        
        for i in tqdm(range(1, len(scale_lines))):
            docParsing = nlp("\n".join(lines[scale_lines[i-1]:scale_lines[i]]))
            for chunk in docParsing.noun_chunks:
                np = removeArticles(chunk.text.lower())
                if np in stopWords:
                    continue
                if np in AllTerms.keys():
                    AllTerms[np] += 1
                else:
                    AllTerms[np] = 1

    return AllTerms

if __name__ == '__main__':
    main()
