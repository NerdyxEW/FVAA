# python3 knn_gpu.py ../../resources/data/small.arff --k 3
# python3 knn_gpu.py ../../resources/data/small.arff --k 3 --output gpu_knn_results.txt
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PA1'))
from kNN import (
    getArffData,
    getUniqueClasses,
    buildConfusionMatrix,
    formatConfusionMatrix,
    euclideanDistance,
    getAllDistances,
    getKNearest,
    predictLabel,
)

try:
    import cupy as cp
    gpuReady = True
except ImportError:
    gpuReady = False

scriptDir = os.path.dirname(os.path.abspath(__file__))
outputPath = os.path.join(scriptDir, 'gpu_knn_results.txt')
useGpu = True
cpuElapsed = 0.0
gpuElapsed = 0.0

def parseArgs(argv):
    if len(argv) < 4:
        print("command: python3 knn_gpu.py <arff_file> --k <int> [--output <path>]")
        sys.exit(1)
    arffPath = argv[1]
    kVal = None
    outPath = outputPath
    i = 2
    while i < len(argv):
        if argv[i] == '--k' and i + 1 < len(argv):
            kVal = int(argv[i + 1])
            i += 2
        elif argv[i] == '--output' and i + 1 < len(argv):
            outPath = argv[i + 1]
            i += 2
        else:
            print(f"error")
            sys.exit(1)
    if kVal is None:
        sys.exit(1)
    return arffPath, kVal, outPath

def moveToGpu(featureList):
    if not gpuReady:
        return featureList
    return cp.asarray(featureList)

def moveFromGpu(gpuArr):
    if not gpuReady:
        return gpuArr
    return cp.asnumpy(gpuArr)

def computeGpuDistances(testIdx, gpuFeatures):
    testArr = gpuFeatures[testIdx]
    dists = cp.sqrt(cp.sum((gpuFeatures - testArr) ** 2, axis=1))
    distList = moveFromGpu(dists)
    pairs = []
    trainIdx = 0
    for j in range(len(distList)):
        if j == testIdx:
            continue
        pairs.append((float(distList[j]), trainIdx))
        trainIdx += 1
    return pairs

def pickTopK(distances, kVal, trainLabels):
    effectiveK = min(kVal, len(distances))
    kNearest = getKNearest(distances, effectiveK)
    predicted, _ = predictLabel(kNearest, trainLabels)
    return predicted

def runGpuLeaveOneOut(features, labels, kVal):
    predictions = []
    gpuFeatures = moveToGpu(features) if gpuReady and useGpu else None
    for i in range(len(features)):
        actualLabel = labels[i]
        trainLabs = [labels[j] for j in range(len(labels)) if j != i]
        if gpuFeatures is not None:
            dists = computeGpuDistances(i, gpuFeatures)
        else:
            testPoint = features[i]
            trainFeats = [features[j] for j in range(len(features)) if j != i]
            dists = [(euclideanDistance(testPoint, f), idx) for idx, f in enumerate(trainFeats)]
        predicted = pickTopK(dists, kVal, trainLabs)
        predictions.append((actualLabel, predicted))
    return predictions

def runCpuBaseline(features, labels, kVal):
    distFunc = euclideanDistance
    predictions = []
    for i in range(len(features)):
        testPoint = features[i]
        actualLabel = labels[i]
        trainFeats = [features[j] for j in range(len(features)) if j != i]
        trainLabs = [labels[j] for j in range(len(labels)) if j != i]
        effectiveK = min(kVal, len(trainFeats))
        allDists = getAllDistances(testPoint, trainFeats, distFunc)
        kNearest = getKNearest(allDists, effectiveK)
        predicted, _ = predictLabel(kNearest, trainLabs)
        predictions.append((actualLabel, predicted))
    return predictions

def compareCpuGpu(cpuPreds, gpuPreds):
    matches = 0
    total = len(cpuPreds)
    for i in range(total):
        if cpuPreds[i][1] == gpuPreds[i][1]:
            matches += 1
    return matches, total

def verifyResults(cpuPreds, gpuPreds, cpuTime, gpuTime):
    matchCount, total = compareCpuGpu(cpuPreds, gpuPreds)
    if matchCount != total:
        print(f"verification failed: {matchCount}/{total} predictions match cpu")
        sys.exit(1)
    if cpuTime <= 0 or gpuTime <= 0:
        print("verification failed: invalid timing")
        sys.exit(1)
    return matchCount, total, cpuTime / gpuTime

def buildGpuReport(arffPath, kVal, classes, cpuTime, gpuTime, cpuPreds, gpuPreds, speedup):
    gpuMatrix = buildConfusionMatrix(gpuPreds, classes)
    cpuMatrix = buildConfusionMatrix(cpuPreds, classes)
    matchCount, total = compareCpuGpu(cpuPreds, gpuPreds)

    gpuCorrect = sum(gpuMatrix[c][c] for c in classes)
    cpuCorrect = sum(cpuMatrix[c][c] for c in classes)

    lines = [
        f'arff file: {arffPath}',
        f'k: {kVal}',
        f'gpu enabled: {gpuReady and useGpu}',
        f'cpu operation time: {cpuTime:.6f} seconds',
        f'gpu operation time: {gpuTime:.6f} seconds',
        f'predictions matching cpu: {matchCount}/{total}',
        f'verification: passed',
        f'speedup (cpu time / gpu time): {speedup:.4f}x',
        '',
        'gpu confusion matrix (rows=actual, columns=predicted):',
        formatConfusionMatrix(gpuMatrix, classes),
        '',
        'compiled metrics (gpu):',
        f'accuracy: {gpuCorrect}/{total} ({gpuCorrect / total if total else 0:.4f})',
        '',
        'compiled metrics (cpu baseline):',
        f'accuracy: {cpuCorrect}/{total} ({cpuCorrect / total if total else 0:.4f})',
        '',
    ]
    return '\n'.join(lines)

def writeResults(reportText, outPath):
    with open(outPath, 'w') as f:
        f.write(reportText + '\n')

if __name__ == '__main__':
    arffPath, kVal, outPath = parseArgs(sys.argv)
    features, labels = getArffData(arffPath)
    classes = getUniqueClasses(labels)

    cpuStart = time.time()
    cpuPreds = runCpuBaseline(features, labels, kVal)
    cpuElapsed = time.time() - cpuStart

    gpuStart = time.time()
    gpuPreds = runGpuLeaveOneOut(features, labels, kVal)
    gpuElapsed = time.time() - gpuStart

    matchCount, total, speedup = verifyResults(cpuPreds, gpuPreds, cpuElapsed, gpuElapsed)

    report = buildGpuReport(arffPath, kVal, classes, cpuElapsed, gpuElapsed, cpuPreds, gpuPreds, speedup)
    writeResults(report, outPath)
    print(report)
    print(f'\nresults written to: {outPath}')
