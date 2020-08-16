import os
import sys

fin = open(sys.argv[1], "r")
fout = open(sys.argv[2], "w")

while True:
	line_Pid = fin.readline()
	line_Pseq = fin.readline()
	line_Pseq = line_Pseq.rstrip('\n').rstrip(' ')
	if not line_Pseq:
		break
		
	for i in range(0,len(line_Pseq)):
		fout.write(line_Pseq[i])
		fout.write(' ')

	fout.write('\n')
	
fin.close()
fout.close()