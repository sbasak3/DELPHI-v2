from bio_embeddings.embed import AlbertEmbedder
from pathlib import Path
import os
import sys


def compute_albert(seq):
	model_dir = '/home/sbasak3/DELPHI/bio_embedding_models/albert'

	embedder = AlbertEmbedder(model_directory = model_dir)

	seq = seq.rstrip('\n').rstrip(' ')
	
	# List-of-Lists with shape [L,4096]
	embedding = embedder.embed(seq)

	albert = []
	for i in range(0,len(seq)):
		value = 0
		for j in range(0,4096):
			value = value + embedding[i][j]
		albert.append(value)
			
	albert_norm = []
	for i in range(0,len(albert)):
		norm_val = (albert[i] - min(albert))/(max(albert) - min(albert))
		albert_norm.append(norm_val)
		
	return albert_norm

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
        feature = compute_albert(line_Pseq)
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