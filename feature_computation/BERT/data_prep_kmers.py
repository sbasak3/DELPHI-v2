import os
import sys
import json

fin = open(sys.argv[1], "r")
fout = open(sys.argv[2], "w")

n = 3

while True:
	line_Pid = fin.readline()
	line_Pseq = fin.readline()
	line_Pseq = line_Pseq.rstrip('\n').rstrip(' ')
	if not line_Pseq:
		break

	grams = [line_Pseq[k:k+n] for k in range(0, len(line_Pseq))]
	for i in range(0,len(grams)):
		fout.write(grams[i])
		fout.write(' ')
	fout.write('\n')
	
fin.close()
fout.close()