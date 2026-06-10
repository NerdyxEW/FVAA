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

def getUniqueClasses(labels): return sorted(set(labels))

def buildConfusionMatrix(predictions, classes):
    matrix = {actual: {pred: 0 for pred in classes} for actual in classes}
    for actual, predicted in predictions:
        matrix[actual][predicted] += 1
    return matrix

def formatConfusionMatrix(matrix, classes):
    labelWidth = max(len(c) for c in classes + ['actual \\ pred'])
    colWidth = max(6, max(len(c) for c in classes))
    header = ' ' * (labelWidth + 2) + '  '.join(c.rjust(colWidth) for c in classes)
    lines = [header]
    for actual in classes:
        row = actual.ljust(labelWidth) + '  ' + '  '.join(str(matrix[actual][pred]).rjust(colWidth) for pred in classes)
        lines.append(row)
    return '\n'.join(lines)

def buildResultsReport(arffPath, distName, distFlag, kVal, pVal, elapsed, matrix, classes):
    pLine = f"p (minkowski exponent): {pVal}" if distFlag == 3 else f"p (minkowski exponent): n/a (not used for {distName})"
    return '\n'.join([
        f"arff file: {arffPath}",
        f"distance metric: {distName} ({distFlag})",
        f"k (nearest neighbors): {kVal}",
        pLine,
        f"operation time: {elapsed:.6f} seconds",
        "",
        "confusion matrix (rows=actual, columns=predicted):",
        formatConfusionMatrix(matrix, classes),
        "",
        "compiled metrics:",
        formatMetrics(matrix, classes),
    ])

def formatMetrics(matrix, classes):
    def div(n, d): return n / d if d else 0.0

    total = sum(matrix[a][p] for a in classes for p in classes)
    correct = sum(matrix[c][c] for c in classes)
    lines = [f"accuracy: {correct}/{total} ({div(correct, total):.4f})", "per-class metrics:"]
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = matrix[c][c]
        fp = sum(matrix[a][c] for a in classes) - tp
        fn = sum(matrix[c][p] for p in classes) - tp
        p, r = div(tp, tp + fp), div(tp, tp + fn)
        f = div(2 * p * r, p + r)
        precisions.append(p); recalls.append(r); f1s.append(f)
        lines.append(f"  {c}: precision={p:.4f}, recall={r:.4f}, f1={f:.4f}")
    n = len(classes) or 1
    lines.append(
        f"macro avg: precision={sum(precisions) / n:.4f}, "
        f"recall={sum(recalls) / n:.4f}, f1={sum(f1s) / n:.4f}"
    )
    return '\n'.join(lines)

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
        predicted, _ = predictLabel(kNearest, trainLabels)
        predictions.append((actual, predicted))
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
    print("  --output     : results file path (default: knn_results.txt)")

if len(sys.argv) < 6:
    printUsage()
    sys.exit(1)

arffPath, distFlag, kVal, pVal, outputPath = sys.argv[1], None, None, 3.0, 'knn_results.txt'
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
    elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
        outputPath = sys.argv[i + 1]
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

features,labels = getArffData(arffPath)

startTime = time.time()
predictions = runLeaveOneOut(features, labels, kVal, distFunc)
endTime = time.time()

classes = getUniqueClasses(labels)
confusionMatrix = buildConfusionMatrix(predictions, classes)
elapsed = endTime - startTime
resultsReport = buildResultsReport(arffPath, distName, distFlag, kVal, pVal, elapsed, confusionMatrix, classes)

with open(outputPath, 'w') as file: file.write(resultsReport + '\n')

print(resultsReport)
print(f"\nresults written to: {outputPath}")