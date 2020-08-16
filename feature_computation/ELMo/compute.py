from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import os
import sys


def compute_seqvec(seq):
	model_dir = Path('/home/sbasak3/DELPHI/bio_embedding_models/ELMo')
	weights = model_dir / 'weights.hdf5'
	options = model_dir / 'options.json'

	# cuda_device=-1 for CPU
	embedder = ElmoEmbedder(options,weights, cuda_device=0)

	seq = seq.rstrip('\n').rstrip(' ')
	
	# List-of-Lists with shape [3,L,1024]
	embedding = embedder.embed_sentence(list(seq))

	seqvec = []
	#for i in range(0,len(seq)):
		#value = 0
		#for j in range(0,1024):
			#value = value + embedding[2][i][j]
		#seqvec.append(value)

	for i in range(0,1024):
		tmp = []
		for j in range(0,len(seq)):
			value = embedding[2][j][i]
			tmp.append(value)
		seqvec.append(tmp)
			
	seqvec_norm = []
	#max = max(seqvec)
	#min = min(seqvec)
	#for i in range(0,len(seqvec)):
		#norm_val = (seqvec[i] - min(seqvec))/(max(seqvec) - min(seqvec))
		#seqvec_norm.append(norm_val)

	for i in range(0,len(seqvec)):
		tmp_norm = []
		for j in range(0,len(seqvec[i])):
			norm_val = (seqvec[i][j] - min(seqvec[i]))/(max(seqvec[i]) - min(seqvec[i]))
			tmp_norm.append(norm_val)
		seqvec_norm.append(tmp_norm)
		
	return seqvec_norm

def load_fasta_and_compute(seq_fn, out_fn):
    fin = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_Pid = fin.readline()
        line_Pseq = fin.readline()
        #line_Plabel = fin.readline()
        if not line_Pseq:
            break
        fout.write(line_Pid)
        fout.write(line_Pseq)
        feature = compute_seqvec(line_Pseq)
        #fout.write(",".join(map(str,feature)) + "\n")
        for i in range(0,len(feature)):
        	fout.write(",".join(map(str,feature[i])) + "\n")
    fin.close()
    fout.close()

	
def main():
	# Input amino acid sequence
	seq_fn = sys.argv[1]
	out_fn = sys.argv[2]
	load_fasta_and_compute(seq_fn, out_fn)
	
if __name__ == '__main__':
    main()