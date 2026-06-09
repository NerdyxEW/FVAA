# python3 kNN.py ../../resources/data/small.arff --distance 1 --k 3
# python3 kNN.py ../../resources/data/small.arff --distance 3 --k 3 --p 4
import math
import sys
import time

def euclideanDistance(a, b): return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def manhattanDistance(a, b): return sum(abs(a[i] - b[i]) for i in range(len(a)))

def minkowskiDistance(a, b, p): return sum(abs(a[i] - b[i]) ** p for i in range(len(a))) ** (1 / p)

def getAllDistances(testPoint, features, distFunc): return [(distFunc(testPoint, feature), idx) for idx, feature in enumerate(features)]

def getKNearest(distances, k): return sorted(distances, key=lambda d: d[0])[:k]

def getArffData(filename):
    features,labels = [],[]
    data = False
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().lower()
            if not line: continue
            if line.startswith('@data'): data = True; continue
            if line.startswith('@'): continue
            if data:
                p = line.split(',')
                features.append([float(v) for v in p[:-1]])
                labels.append(p[-1].strip())
    return features,labels

def printUsage():
    print("command: python3 kNN.py <arff_file> --distance <1 or 2 or 3> --k <int> [--p <float>]")
    print("  <arff_file>  : path to training data in ARFF format")
    print("  --distance   : distance metric (1=euclidean, 2=manhattan, 3=minkowski)")
    print("  --k          : number of nearest neighbors to select")
    print("  --p          : minkowski exponent (only used when --distance 3; default 3.0)")

if len(sys.argv) < 6:
    printUsage()
    sys.exit(1)

arffPath, distFlag, kVal, pVal = sys.argv[1], None, None, 3.0
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
        print(f"error, unknown or incomplete argument: {sys.argv[i]}")
        printUsage()
        sys.exit(1)

if distFlag not in [1, 2, 3] or kVal is None:
    print("missing or invalid --distance (1|2|3) and/or --k <int>")
    printUsage()
    sys.exit(1)

distName = 'euclidean' if distFlag == 1 else 'manhattan' if distFlag == 2 else 'minkowski'
pVal = 2.0 if distFlag == 1 else 1.0 if distFlag == 2 else pVal
distFunc = (
    euclideanDistance if distFlag == 1
    else manhattanDistance if distFlag == 2
    else lambda a, b: minkowskiDistance(a, b, pVal)
)

print(f"arff file: {arffPath}")
print(f"distance metric: {distName} ({distFlag})")
print(f"k (nearest neighbors): {kVal}")
print(f"p (minkowski exponent): {pVal}" if distFlag == 3 else f"p (minkowski exponent): n/a (not used for {distName})")

features,labels = getArffData(arffPath)

testPoint = [2.1, 3.2, 4.3, 5.4]

matrix = {actual: {pred: 0 for pred in labels} for actual in labels} # unfinished

startTime = time.time()

allDistances = getAllDistances(testPoint, features, distFunc)
kNearest = getKNearest(allDistances, kVal)

print(f"\ntest point: {testPoint}")
print(f"all distances ({len(allDistances)} points):")
for dist, idx in sorted(allDistances, key=lambda d: d[0]):
    print(f"  idx {idx}: distance={dist:.4f}, label={labels[idx]}, features={features[idx]}")

print(f"\nclosest {kVal} neighbor(s):")
for dist, idx in kNearest:
    print(f"  idx {idx}: distance={dist:.4f}, label={labels[idx]}, features={features[idx]}")

endTime = time.time()
print(f"\noperation time: {endTime - startTime:.6f} seconds")
