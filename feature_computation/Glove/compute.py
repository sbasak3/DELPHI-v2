from bio_embeddings.embed import GloveEmbedder
from pathlib import Path
import os
import sys


def compute_glove(seq):
	model_fn = '/home/sbasak3/DELPHI/bio_embedding_models/glove/glove.model'

	embedder = GloveEmbedder(model_file = model_fn)

	seq = seq.rstrip('\n').rstrip(' ')
	
	# List-of-Lists with shape [L,512]
	embedding = embedder.embed(seq)

	glove = []
	for i in range(0,len(seq)):
		value = 0
		for j in range(0,512):
			value = value + embedding[i][j]
		glove.append(value)
			
	glove_norm = []
	for i in range(0,len(glove)):
		norm_val = (glove[i] - min(glove))/(max(glove) - min(glove))
		glove_norm.append(norm_val)
		
	return glove_norm

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
        feature = compute_glove(line_Pseq)
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