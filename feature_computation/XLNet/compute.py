from bio_embeddings.embed import XLNetEmbedder
from pathlib import Path
import os
import sys


def compute_xlnet(seq):
	model_dir = '/home/sbasak3/DELPHI/bio_embedding_models/xlnet'

	embedder = XLNetEmbedder(model_directory = model_dir)

	seq = seq.rstrip('\n').rstrip(' ')
	
	# List-of-Lists with shape [L,1024]
	embedding = embedder.embed(seq)

	xlnet = []
	for i in range(0,len(seq)):
		value = 0
		for j in range(0,1024):
			value = value + embedding[i][j]
		xlnet.append(value)
			
	xlnet_norm = []
	for i in range(0,len(xlnet)):
		norm_val = (xlnet[i] - min(xlnet))/(max(xlnet) - min(xlnet))
		xlnet_norm.append(norm_val)
		
	return xlnet_norm

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
        feature = compute_xlnet(line_Pseq)
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