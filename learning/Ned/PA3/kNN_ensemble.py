# uv run kNN_ensemble.py ../../resources/data/small.arff --k 3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PA1'))
from kNN import (
    getArffData,
    getUniqueClasses,
    getDistFunc,
    runLeaveOneOut,
    buildConfusionMatrix,
    formatConfusionMatrix,
)

scriptDir = os.path.dirname(os.path.abspath(__file__))
outputPath = os.path.join(scriptDir, 'output_ensemble.md')
classifierConfigs = [(1, 3.0), (2, 3.0), (3, 3.0)]

def parseArgs(argv):
    if len(argv) < 4:
        print("command: uv run kNN_ensemble.py <arff_file> --k <int>")
        sys.exit(1)
    arffPath = argv[1]
    kVal = None
    i = 2
    while i < len(argv):
        if argv[i] == '--k' and i + 1 < len(argv):
            kVal = int(argv[i + 1])
            i += 2
        else:
            print(f"error, unknown argument: {argv[i]}")
            sys.exit(1)
    if kVal is None:
        print("missing --k <int>")
        sys.exit(1)
    return arffPath, kVal

def getAccuracy(matrix, classes):
    total = sum(matrix[a][p] for a in classes for p in classes)
    correct = sum(matrix[c][c] for c in classes)
    return correct, total, correct / total if total else 0.0

def runEnsemble(features, labels, kVal, classes):
    results = []
    for distFlag, pVal in classifierConfigs:
        distName, pVal, distFunc = getDistFunc(distFlag, pVal)
        preds = runLeaveOneOut(features, labels, kVal, distFunc)
        matrix = buildConfusionMatrix(preds, classes)
        results.append({
            'distName': distName,
            'distFlag': distFlag,
            'pVal': pVal,
            'predictions': preds,
            'matrix': matrix,
        })
    return results

def combinePredictions(classifierResults):
    sampleCount = len(classifierResults[0]['predictions'])
    ensemblePreds = []
    for i in range(sampleCount):
        actual = classifierResults[0]['predictions'][i][0]
        votes = {}
        for result in classifierResults:
            label = result['predictions'][i][1]
            votes[label] = votes.get(label, 0) + 1
        maxVotes = max(votes.values())
        tied = [label for label, count in votes.items() if count == maxVotes]
        if len(tied) == 1:
            predicted = tied[0]
        else:
            predicted = None
            for result in classifierResults:
                label = result['predictions'][i][1]
                if label in tied:
                    predicted = label
                    break
        ensemblePreds.append((actual, predicted))
    return ensemblePreds

def writeOutput(arffPath, kVal, classes, classifierResults, ensemblePreds):
    lines = [
        '# kNN Ensemble Report',
        '',
        f'arff file: {arffPath}',
        f'k: {kVal}',
        '',
        '## Individual Classifiers',
        '',
    ]
    for result in classifierResults:
        correct, total, accuracy = getAccuracy(result['matrix'], classes)
        pLine = f' (p={result["pVal"]})' if result['distFlag'] == 3 else ''
        lines.extend([
            f'### {result["distName"]} - distance {result["distFlag"]}{pLine}',
            '',
            '```',
            formatConfusionMatrix(result['matrix'], classes),
            '```',
            '',
            f'accuracy: {correct}/{total} ({accuracy:.4f})',
            '',
        ])
    ensembleMatrix = buildConfusionMatrix(ensemblePreds, classes)
    correct, total, accuracy = getAccuracy(ensembleMatrix, classes)
    lines.extend([
        '## Ensemble',
        '',
        'combination: majority vote across euclidean, manhattan, and minkowski classifiers',
        '',
        '```',
        formatConfusionMatrix(ensembleMatrix, classes),
        '```',
        '',
        f'accuracy: {correct}/{total} ({accuracy:.4f})',
        '',
    ])
    with open(outputPath, 'w') as file:
        file.write('\n'.join(lines))

if __name__ == '__main__':
    arffPath, kVal = parseArgs(sys.argv)
    features, labels = getArffData(arffPath)
    classes = getUniqueClasses(labels)
    classifierResults = runEnsemble(features, labels, kVal, classes)
    ensemblePreds = combinePredictions(classifierResults)
    writeOutput(arffPath, kVal, classes, classifierResults, ensemblePreds)
    print(f'results written to: {outputPath}')
