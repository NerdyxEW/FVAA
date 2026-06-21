# python3 kNN_report.py ../../resources/data/small.arff --distance 1 --k 3
# python3 kNN_report.py ../../resources/data/medium.arff --distance 3 --k 3 --p 4
import os
import sys
import time

def getPeakMemoryMb():
    try:
        import resource
        peakKb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peakKb / 1024
    except ImportError:
        import tracemalloc
        currentBytes, peakBytes = tracemalloc.get_traced_memory()
        return peakBytes / (1024 * 1024)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PA1'))
from kNN import (
    buildConfusionMatrix,
    formatConfusionMatrix,
    getArffData,
    getDistFunc,
    getUniqueClasses,
    parseArgs,
    printUsage,
    runLeaveOneOut,
)

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

def buildDetailedReport(arffPath, distName, distFlag, kVal, pVal, elapsed, peakMemoryMb, matrix, classes):
    pLine = f"p (minkowski exponent): {pVal}" if distFlag == 3 else f"p (minkowski exponent): n/a (not used for {distName})"
    return '\n'.join([
        f"arff file: {arffPath}",
        f"distance metric: {distName} ({distFlag})",
        f"k (nearest neighbors): {kVal}",
        pLine,
        f"operation time: {elapsed:.6f} seconds",
        f"peak memory: {peakMemoryMb:.2f} MB",
        "",
        "confusion matrix (rows=actual, columns=predicted):",
        formatConfusionMatrix(matrix, classes),
        "",
        "compiled metrics:",
        formatMetrics(matrix, classes),
    ])

if __name__ == '__main__':
    if len(sys.argv) < 6:
        printUsage()
        sys.exit(1)

    arffPath, distFlag, kVal, pVal, outputPath = parseArgs(sys.argv)
    if outputPath == 'knn_results.txt':
        outputPath = 'knn_report.txt'

    distName, pVal, distFunc = getDistFunc(distFlag, pVal)
    features, labels = getArffData(arffPath)

    try:
        import resource
    except ImportError:
        import tracemalloc
        tracemalloc.start()

    startTime = time.time()
    predictions = runLeaveOneOut(features, labels, kVal, distFunc)
    endTime = time.time()
    peakMemoryMb = getPeakMemoryMb()

    classes = getUniqueClasses(labels)
    confusionMatrix = buildConfusionMatrix(predictions, classes)
    elapsed = endTime - startTime
    resultsReport = buildDetailedReport(
        arffPath, distName, distFlag, kVal, pVal, elapsed, peakMemoryMb, confusionMatrix, classes
    )

    with open(outputPath, 'w') as file:
        file.write(resultsReport + '\n')

    print(resultsReport)
    print(f"\nresults written to: {outputPath}")
