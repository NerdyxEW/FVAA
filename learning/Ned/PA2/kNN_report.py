# python3 kNN_report.py ../../resources/data/small.arff --distance 1 --k 3
# python3 kNN_report.py ../../resources/data/medium.arff --distance 3 --k 3 --p 4
import os
import sys
import time

import matplotlib.pyplot as plt

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

def getClassMetrics(matrix, classes):
    def div(n, d): return n / d if d else 0.0

    total = sum(matrix[a][p] for a in classes for p in classes)
    correct = sum(matrix[c][c] for c in classes)
    accuracy = div(correct, total)
    classRows = []
    for c in classes:
        truePositives = matrix[c][c]
        falsePositives = sum(matrix[a][c] for a in classes) - truePositives
        falseNegatives = sum(matrix[c][p] for p in classes) - truePositives
        trueNegatives = total - truePositives - falsePositives - falseNegatives
        precision = div(truePositives, truePositives + falsePositives)
        recall = div(truePositives, truePositives + falseNegatives)
        f1Score = div(2 * precision * recall, precision + recall)
        sensitivity = div(truePositives, truePositives + falseNegatives)
        specificity = div(trueNegatives, trueNegatives + falsePositives)
        classRows.append({
            'className': c,
            'precision': precision,
            'recall': recall,
            'f1Score': f1Score,
            'sensitivity': sensitivity,
            'specificity': specificity,
        })
    return accuracy, correct, total, classRows

def formatMetrics(matrix, classes):
    accuracy, correct, total, classRows = getClassMetrics(matrix, classes)
    lines = [f"accuracy: {correct}/{total} ({accuracy:.4f})", "per-class metrics:"]
    precisions, recalls, f1s, sensitivities, specificities = [], [], [], [], []
    for row in classRows:
        precisions.append(row['precision']); recalls.append(row['recall']); f1s.append(row['f1Score'])
        sensitivities.append(row['sensitivity']); specificities.append(row['specificity'])
        lines.append(
            f"  {row['className']}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"f1={row['f1Score']:.4f}, sensitivity={row['sensitivity']:.4f}, specificity={row['specificity']:.4f}"
        )
    n = len(classes) or 1
    lines.append(
        f"macro avg: precision={sum(precisions) / n:.4f}, "
        f"recall={sum(recalls) / n:.4f}, f1={sum(f1s) / n:.4f}, "
        f"sensitivity={sum(sensitivities) / n:.4f}, specificity={sum(specificities) / n:.4f}"
    )
    return '\n'.join(lines)

def saveConfusionMatrixPlot(matrix, classes, filePath):
    rowCount = len(classes)
    cellValues = [[matrix[actual][pred] for pred in classes] for actual in classes]
    figure, axis = plt.subplots()
    axis.imshow(cellValues, cmap='Blues')
    axis.set_xticks(range(rowCount))
    axis.set_yticks(range(rowCount))
    axis.set_xticklabels(classes)
    axis.set_yticklabels(classes)
    axis.set_xlabel('predicted')
    axis.set_ylabel('actual')
    axis.set_title('confusion matrix')
    for rowIndex in range(rowCount):
        for colIndex in range(rowCount):
            axis.text(colIndex, rowIndex, str(cellValues[rowIndex][colIndex]), ha='center', va='center')
    figure.tight_layout()
    figure.savefig(filePath)
    plt.close(figure)

def saveMetricsTablePlot(matrix, classes, filePath):
    accuracy, correct, total, classRows = getClassMetrics(matrix, classes)
    tableHeaders = ['class', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']
    tableRows = [
        [
            row['className'],
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1Score']:.4f}",
            f"{row['sensitivity']:.4f}",
            f"{row['specificity']:.4f}",
        ]
        for row in classRows
    ]
    tableRows.append(['accuracy', f"{accuracy:.4f}", '', '', '', ''])
    figure, axis = plt.subplots()
    axis.axis('off')
    axis.set_title('metrics table')
    table = axis.table(cellText=tableRows, colLabels=tableHeaders, loc='center', cellLoc='center')
    table.scale(1.2, 1.4)
    figure.tight_layout()
    figure.savefig(filePath)
    plt.close(figure)

def buildDetailedReport(arffPath, distName, distFlag, kVal, pVal, elapsed, peakMemoryMb, matrix, classes, confusionPlotPath, metricsPlotPath):
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
        "",
        "visualizations:",
        f"confusion matrix plot: {confusionPlotPath}",
        f"metrics table plot: {metricsPlotPath}",
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

    outputDir = os.path.dirname(outputPath) or '.'
    dataBaseName = os.path.splitext(os.path.basename(arffPath))[0]
    confusionPlotPath = os.path.join(outputDir, f'confusion_matrix_{dataBaseName}.png')
    metricsPlotPath = os.path.join(outputDir, f'metrics_table_{dataBaseName}.png')
    saveConfusionMatrixPlot(confusionMatrix, classes, confusionPlotPath)
    saveMetricsTablePlot(confusionMatrix, classes, metricsPlotPath)

    resultsReport = buildDetailedReport(
        arffPath, distName, distFlag, kVal, pVal, elapsed, peakMemoryMb,
        confusionMatrix, classes, confusionPlotPath, metricsPlotPath
    )

    with open(outputPath, 'w') as file:
        file.write(resultsReport + '\n')

    print(resultsReport)
    print(f"\nresults written to: {outputPath}")
    print(f"confusion matrix plot saved to: {confusionPlotPath}")
    print(f"metrics table plot saved to: {metricsPlotPath}")
