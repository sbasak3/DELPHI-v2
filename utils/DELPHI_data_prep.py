import os
import sys

fin = open(sys.argv[1], "r")
fout = open(sys.argv[2], "w")


while True:
	line_Pid = fin.readline()
	line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
	#line_Pseq = fin.readline().rstrip(' ')
	line_Plabel = fin.readline()
	#line_feature = fin.readline().rstrip(' ')
	if not line_Pseq:
		break
		
	fout.write(line_Pid)
	fout.write(line_Pseq)
	fout.write('\n')
	#if len(line_Pseq) < 2048:
		#fout.write(line_Pid)
		#fout.write(line_Pseq)
		#fout.write(line_feature)
	
fin.close()
fout.close()