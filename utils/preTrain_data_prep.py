import pandas as pd

seq_file = 'uniprot-reviewed.xlsx'
data = pd.read_excel(seq_file,sheet_name='Sheet0',header=0,index_col=False,keep_default_na=True)
out_seq_file = open("train_sequence.txt", "a")
vocab_file = open("vocab.txt", "a")

n = 5
vocab = {}

vocab['[PAD]'] = 0
vocab['[CLS]'] = 0
vocab['[SEP]'] = 0
vocab['[MASK]'] = 0

for i in range(0,len(data['Sequence'])):
	seq = data['Sequence'][i]
	grams = [seq[k:k+n] for k in range(0, len(seq))]
	for j in range(0,len(grams)):
		if grams[j] in vocab.keys():
			vocab[grams[j]] = vocab[grams[j]] + 1
		else:
			vocab[grams[j]] = 0
		out_seq_file.write(grams[j])
		out_seq_file.write(' ')
	out_seq_file.write('\n')

vocab['[UNK]'] = 0

for i in vocab.keys():
	vocab_file.write(i)
	vocab_file.write('\n')
	
vocab_file.close()
out_seq_file.close()