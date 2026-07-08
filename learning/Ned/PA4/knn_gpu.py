# python3 knn_gpu.py ../../resources/data/small.arff --k 3
# python3 knn_gpu.py --benchmark --k 3
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
dataDir = os.path.join(scriptDir, '..', '..', 'resources', 'data')
outputPath = os.path.join(scriptDir, 'gpu_knn_results.txt')
benchmarkPath = os.path.join(scriptDir, 'benchmark_results.txt')
useGpu = True

def parseArgs(argv):
    if len(argv) < 2:
        print("command: python3 knn_gpu.py <arff_file> --k <int> [--output <path>]")
        print("         python3 knn_gpu.py --benchmark --k <int>")
        sys.exit(1)

    kVal = None
    outPath = outputPath
    benchmarkMode = False
    arffPath = None
    i = 1

    while i < len(argv):
        if argv[i] == '--benchmark':
            benchmarkMode = True
            i += 1
        elif argv[i] == '--k' and i + 1 < len(argv):
            kVal = int(argv[i + 1])
            i += 2
        elif argv[i] == '--output' and i + 1 < len(argv):
            outPath = argv[i + 1]
            i += 2
        elif not argv[i].startswith('--'):
            arffPath = argv[i]
            i += 1
        else:
            print(f"error")
            sys.exit(1)

    if kVal is None:
        print("missing --k <int>")
        sys.exit(1)
    if not benchmarkMode and arffPath is None:
        print("missing arff file or --benchmark")
        sys.exit(1)
    return arffPath, kVal, outPath, benchmarkMode

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
    predictions = []
    for i in range(len(features)):
        testPoint = features[i]
        actualLabel = labels[i]
        trainFeats = [features[j] for j in range(len(features)) if j != i]
        trainLabs = [labels[j] for j in range(len(labels)) if j != i]
        effectiveK = min(kVal, len(trainFeats))
        allDists = getAllDistances(testPoint, trainFeats, euclideanDistance)
        kNearest = getKNearest(allDists, effectiveK)
        predicted, _ = predictLabel(kNearest, trainLabs)
        predictions.append((actualLabel, predicted))
    return predictions

def compareCpuGpu(cpuPreds, gpuPreds):
    matches = sum(1 for i in range(len(cpuPreds)) if cpuPreds[i][1] == gpuPreds[i][1])
    return matches, len(cpuPreds)

def verifyResults(cpuPreds, gpuPreds, cpuTime, gpuTime):
    matchCount, total = compareCpuGpu(cpuPreds, gpuPreds)
    if matchCount != total:
        print(f"verification failed: {matchCount}/{total} predictions match cpu")
        sys.exit(1)
    if cpuTime <= 0 or gpuTime <= 0:
        print("verification failed: invalid timing")
        sys.exit(1)
    return matchCount, total, cpuTime / gpuTime

def runDataset(arffPath, kVal):
    features, labels = getArffData(arffPath)
    classes = getUniqueClasses(labels)

    cpuStart = time.time()
    cpuPreds = runCpuBaseline(features, labels, kVal)
    cpuElapsed = time.time() - cpuStart

    gpuStart = time.time()
    gpuPreds = runGpuLeaveOneOut(features, labels, kVal)
    gpuElapsed = time.time() - gpuStart

    matchCount, total, speedup = verifyResults(cpuPreds, gpuPreds, cpuElapsed, gpuElapsed)
    gpuMatrix = buildConfusionMatrix(gpuPreds, classes)
    gpuCorrect = sum(gpuMatrix[c][c] for c in classes)

    return {
        'arffPath': arffPath,
        'sampleCount': total,
        'cpuTime': cpuElapsed,
        'gpuTime': gpuElapsed,
        'speedup': speedup,
        'matchCount': matchCount,
        'accuracy': gpuCorrect / total if total else 0,
        'cpuPreds': cpuPreds,
        'gpuPreds': gpuPreds,
        'classes': classes,
    }

def buildGpuReport(result, kVal):
    gpuMatrix = buildConfusionMatrix(result['gpuPreds'], result['classes'])
    cpuMatrix = buildConfusionMatrix(result['cpuPreds'], result['classes'])
    total = result['sampleCount']
    gpuCorrect = sum(gpuMatrix[c][c] for c in result['classes'])
    cpuCorrect = sum(cpuMatrix[c][c] for c in result['classes'])

    lines = [
        f'arff file: {result["arffPath"]}',
        f'k: {kVal}',
        f'gpu enabled: {gpuReady and useGpu}',
        f'cpu operation time: {result["cpuTime"]:.6f} seconds',
        f'gpu operation time: {result["gpuTime"]:.6f} seconds',
        f'predictions matching cpu: {result["matchCount"]}/{total}',
        f'verification: passed',
        f'speedup (cpu time / gpu time): {result["speedup"]:.4f}x',
        '',
        'gpu confusion matrix (rows=actual, columns=predicted):',
        formatConfusionMatrix(gpuMatrix, result['classes']),
        '',
        'compiled metrics (gpu):',
        f'accuracy: {gpuCorrect}/{total} ({gpuCorrect / total if total else 0:.4f})',
        '',
        'compiled metrics (cpu baseline):',
        f'accuracy: {cpuCorrect}/{total} ({cpuCorrect / total if total else 0:.4f})',
        '',
    ]
    return '\n'.join(lines)

def buildBenchmarkReport(results, kVal):
    lines = [
        '# PA4 GPU kNN Benchmark',
        '',
        f'k: {kVal}',
        f'gpu enabled: {gpuReady and useGpu}',
        '',
        '| dataset | samples | cpu time (s) | gpu time (s) | speedup | accuracy | verified |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in results:
        name = os.path.basename(r['arffPath'])
        lines.append(
            f'| {name} | {r["sampleCount"]} | {r["cpuTime"]:.4f} | {r["gpuTime"]:.4f} | {r["speedup"]:.4f}x | {r["accuracy"]:.4f} | {r["matchCount"]}/{r["sampleCount"]} |'
        )
    lines.extend(['', 'see setup.md for cupy install and answers.md for findings'])
    return '\n'.join(lines)

def writeResults(reportText, outPath):
    with open(outPath, 'w') as f:
        f.write(reportText + '\n')

def runBenchmark(kVal):
    datasets = ['small.arff', 'medium.arff', 'large.arff']
    results = []
    for name in datasets:
        path = os.path.join(dataDir, name)
        print(f'running {name}...')
        results.append(runDataset(path, kVal))
    return results

if __name__ == '__main__':
    arffPath, kVal, outPath, benchmarkMode = parseArgs(sys.argv)

    if benchmarkMode:
        results = runBenchmark(kVal)
        report = buildBenchmarkReport(results, kVal)
        writeResults(report, benchmarkPath)
        print(report)
        print(f'\nbenchmark written to: {benchmarkPath}')
        smallResult = results[0]
        writeResults(buildGpuReport(smallResult, kVal), outputPath)
    else:
        result = runDataset(arffPath, kVal)
        report = buildGpuReport(result, kVal)
        writeResults(report, outPath)
        print(report)
        print(f'\nresults written to: {outPath}')
