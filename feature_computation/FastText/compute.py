from bio_embeddings.embed import FastTextEmbedder
from pathlib import Path
import os
import sys


def compute_fasttext(seq):
	model_fn = '/home/sbasak3/DELPHI/bio_embedding_models/fasttext/fasttext.model'

	embedder = FastTextEmbedder(model_file = model_fn)

	seq = seq.rstrip('\n').rstrip(' ')
	
	# List-of-Lists with shape [L,512]
	embedding = embedder.embed(seq)

	fasttext = []
	for i in range(0,len(seq)):
		value = 0
		for j in range(0,512):
			value = value + embedding[i][j]
		fasttext.append(value)
			
	fasttext_norm = []
	for i in range(0,len(fasttext)):
		norm_val = (fasttext[i] - min(fasttext))/(max(fasttext) - min(fasttext))
		fasttext_norm.append(norm_val)
		
	return fasttext_norm

def load_fasta_and_compute(seq_fn, out_fn):
    fin = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_Pid = fin.readline()
        line_Pseq = fin.readline()
        if not line_Pseq:
            break
        fout.write(line_Pid)
        fout.write(line_Pseq)
        feature = compute_fasttext(line_Pseq)
        fout.write(",".join(map(str,feature)) + "\n")
    fin.close()
    fout.close()

	
def main():
	# Input amino acid sequence
	seq_fn = sys.argv[1]
	out_fn = sys.argv[2]
	load_fasta_and_compute(seq_fn, out_fn)
	
if __name__ == '__main__':
    main()