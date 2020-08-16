import sys

fin1 = open(sys.argv[1], "r")
fin2 = open(sys.argv[2], "r")
fout = open(sys.argv[3], "w")

while True:
	line_Pid = fin1.readline()
	line_Pseq = fin1.readline()
	line_Feature = fin2.readline()
	if not line_Pseq:
		break
	fout.write(line_Pid)
	fout.write(line_Pseq)
	fout.write(line_Feature)
	#for i in range(0,768):
		#line_Feature = fin2.readline()
		#fout.write(line_Feature)
	
fin1.close()
fin2.close()
fout.close()