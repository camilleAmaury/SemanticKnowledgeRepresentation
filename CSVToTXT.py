from os.path import join
from tqdm import tqdm
# file reader
import csv
# spacy nlp
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

 

def main():
    """
        Function which extracts a csv file as text file and create line with sentences.
    """
    
    # csv reader - extracting document
    r_file = "TP_CS_Abstract.csv"
    corpus = []
    with open('.\csv_docs\{}'.format(r_file)) as csvfile:
        corpus = [str(row[1]) for row in csv.reader(csvfile, delimiter=',', quotechar='|')]
    
    # txt writer and natural language processing
    w_file = r_file.replace(".csv", ".txt")
    f1 = open(join(".\Corpus", w_file), 'w', errors='ignore')
                
    line_writed = 0
    for i in tqdm(range(len(corpus))):
        subcorpus = corpus[i]
        # text processing
        subcorpus = subcorpus.replace('"', "")
        # splitting lines
        lines = subcorpus.split(".")
        for line in lines:
            if line != "" and line != " " and len(line) > 10:
                line_writed += 1
                f1.write(line+".\n")
                
    print("\n > Line writed : {}".format(line_writed))

    f1.close()
    
"""
    Execution
"""
if __name__ == '__main__':
    main()