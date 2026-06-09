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

def getNeighborVotes(kNearest, trainLabels):
    votes = {}
    for _, idx in kNearest:
        label = trainLabels[idx]
        votes[label] = votes.get(label, 0) + 1
    return votes

def predictLabel(kNearest, trainLabels):
    votes = getNeighborVotes(kNearest, trainLabels)
    maxVotes = max(votes.values())
    tied = [label for label, count in votes.items() if count == maxVotes]
    if len(tied) == 1:
        return tied[0], votes
    for _, idx in kNearest:
        if trainLabels[idx] in tied:
            return trainLabels[idx], votes

def runLeaveOneOut(features, labels, k, distFunc):
    predictions = []
    for i in range(len(features)):
        testPoint = features[i]
        actual = labels[i]
        trainFeatures = [features[j] for j in range(len(features)) if j != i]
        trainLabels = [labels[j] for j in range(len(labels)) if j != i]
        effectiveK = min(k, len(trainFeatures))
        allDistances = getAllDistances(testPoint, trainFeatures, distFunc)
        kNearest = getKNearest(allDistances, effectiveK)
        predicted, votes = predictLabel(kNearest, trainLabels)
        predictions.append((actual, predicted, votes))
    return predictions

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

startTime = time.time()
predictions = runLeaveOneOut(features, labels, kVal, distFunc)
endTime = time.time()

correct = sum(1 for actual, predicted, _ in predictions if actual == predicted)
print(f"\nleave-one-out results ({len(predictions)} points):")
for i, (actual, predicted, votes) in enumerate(predictions):
    match = "ok" if actual == predicted else "miss"
    voteStr = ", ".join(f"{label}={count}" for label, count in sorted(votes.items()))
    print(f"  row {i}: actual={actual}, predicted={predicted}, votes=[{voteStr}] [{match}]")
print(f"\ncorrect: {correct}/{len(predictions)}")
print(f"operation time: {endTime - startTime:.6f} seconds")