# python3 kNN.py ../../resources/data/small.arff --distance 1 --k 3
import math
import sys
import time

def getDistance(pointData, distMethod, pVal):
    total = 0
    for i in range(len(pointData[0])):
        total += abs(pointData[0][i] - pointData[1][i]) ** pVal
    return total ** (1 / pVal)

def getArffData(filename):
    features,labels = [],[]
    data = False
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().lower()
            if not line or line.startswith('@'): continue
            if line.startswith('@data'): data = True; continue
            elif line.startswith('@attribute'): continue
            if data:
                p = line.split(',')
                features.append([float(v) for v in p[:-1]])
                labels.append(p[-1].strip())
    return features,labels

if len(sys.argv) < 6: sys.exit(1)

arffPath,distFlag,kVal,pVal = sys.argv[1],None,None,3.0
i = 2
while i < len(sys.argv):
    if sys.argv[i] == '--distance' and i + 1 < len(sys.argv):
        try: distFlag = int(sys.argv[i + 1])
        except ValueError: distFlag = None
        i += 2
    elif sys.argv[i] == '--k' and i + 1 < len(sys.argv):
        try: kVal = int(sys.argv[i + 1])
        except ValueError: kVal = None
        i += 2
    elif sys.argv[i] == '--p' and i + 1 < len(sys.argv):
        try: pVal = float(sys.argv[i + 1])
        except ValueError: pVal = 3.0
        i += 2
    else:
        print(f"error, missing: {sys.argv[i]}")
        sys.exit(1)

if distFlag not in [1, 2, 3] or kVal is None:
    print("missing --distance 1|2|3 and --k <int>")
    sys.exit(1)

choice = str(distFlag)
if distFlag == 1: pVal = 2.0
elif distFlag == 2: pVal = 1.0

features,labels = getArffData(arffPath)

testPoint = [2.1, 3.2, 4.3, 5.4]
trainPoint = [4.2, 6.4, 8.6, 10.8]

matrix = {actual: {pred: 0 for pred in labels} for actual in labels} # unfinished

startTime = time.time()
print(f"distance: {getDistance([testPoint, trainPoint], choice, pVal)}")

endTime = time.time()
print(f"operation time: {endTime - startTime} seconds")
