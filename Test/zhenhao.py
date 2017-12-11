import numpy as np
point = []
fr = open('jain.csv')
textData = [inst.strip().split(' ') for inst in fr.readlines()]
for line in textData:
    line = map(eval, line)
    point.append(line)
print point[0][0]