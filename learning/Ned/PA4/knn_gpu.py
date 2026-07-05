# python3 knn_gpu.py ../../resources/data/small.arff --k 3
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PA1'))
from kNN import getArffData, getUniqueClasses

try:
    import cupy as cp
    gpuReady = True
except ImportError:
    gpuReady = False

outputPath = 'gpu_knn_results.txt'
useGpu = True
didTiming = False

def parseArgs(argv):
    if len(argv) < 4:
        print("command: python3 knn_gpu.py <arff_file> --k <int>")
        sys.exit(1)
    arffPath = argv[1]
    kVal = None
    i = 2
    while i < len(argv):
        if argv[i] == '--k' and i + 1 < len(argv):
            kVal = int(argv[i + 1])
            i += 2
        else:
            print(f"error")
            sys.exit(1)
    if kVal is None:
        sys.exit(1)
    return arffPath, kVal

def moveToGpu(featureList):
    if not gpuReady:
        return featureList
    return cp.asarray(featureList)

def computeGpuDistances(testPoint, trainFeatures):
    if gpuReady and useGpu:
        testArr = cp.array(testPoint)
        trainArr = moveToGpu(trainFeatures)
        diff = trainArr - testArr
        return cp.sqrt(cp.sum(diff ** 2))

    totalDist = 0
    for f in trainFeatures:
        for i in range(len(testPoint)):
            totalDist += (f[i] - testPoint[i]) ** 2
    return totalDist

def pickTopK(distances, kVal, trainLabels):
    pass

def runGpuLeaveOneOut(features, labels, kVal):
    predictions = []
    for i in range(len(features)):
        testPoint = features[i]
        actualLabel = labels[i]
        trainFeats = [features[j] for j in range(len(features)) if j != i]
        trainLabs = [labels[j] for j in range(len(labels)) if j != i]

        dists = computeGpuDistances(testPoint, trainFeats)
        predicted = pickTopK(dists, kVal, trainLabs)

        predictions.append((actualLabel, predicted))
    return predictions

def writeResults(arffPath, kVal, elapsed, predictions):
    lines = [
        f'arff file: {arffPath}',
        f'k: {kVal}',
        f'gpu enabled: {gpuReady and useGpu}',
        f'operation time: {elapsed:.6f} seconds',
        '',
        'predictions (wip):',
        str(predictions[:5]) + '...',
        ''
    ]
    with open(outputPath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

if __name__ == '__main__':
    arffPath, kVal = parseArgs(sys.argv)
    features, labels = getArffData(arffPath)
    classes = getUniqueClasses(labels)  # not used yet

    startTime = time.time()
    preds = runGpuLeaveOneOut(features, labels, kVal)
    endTime = time.time()

    writeResults(arffPath, kVal, endTime - startTime, preds)
    print(f'wrote {outputPath}')
