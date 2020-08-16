import os
import shutil
import sys

fin = open('/home/sbasak3/DELPHI/Datasets/Dset_355_Pid_Pseq.txt', "r")
source_path = '/home/sbasak3/DELPHI/out/cpu/Dset_448/'
destination_path = '/home/sbasak3/DELPHI/out/cpu/Dset_355/'
extension = '.txt'

files = []

while True:
	line_Pid = fin.readline().lstrip('>').rstrip('\n').rstrip(' ')
	line_Pseq = fin.readline().rstrip('\n').rstrip(' ')
	if not line_Pid:
		break
	file = source_path + line_Pid + extension
	files.append(file)

for f in files:
	shutil.copy(f, destination_path)

fin.close()