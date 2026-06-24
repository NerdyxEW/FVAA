# python3 kNN_ensemble.py ../../resources/data/small.arff --k 3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PA1'))
from kNN import getArffData, getUniqueClasses, parseArgs as parsePa1Args, printUsage as printPa1Usage

outputPath = 'output_ensemble.md'

def parseArgs(argv):
    if len(argv) < 4:
        print("command: python3 kNN_ensemble.py <arff_file> --k <int>")
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

def runEnsemble(features, labels, kVal):
    pass

def combinePredictions():
    pass

def writeOutput(arffPath, kVal, classes):
    lines = [
        '# kNN Ensemble Report',
        '',
        f'arff file: {arffPath}',
        f'k: {kVal}',
        '',
        '## Individual Classifiers',
        '',
        '<!-- TODO -->',
        '',
        '## Ensemble',
        '',
        '<!-- TODO -->',
    ]
    with open(outputPath, 'w') as file:
        file.write('\n'.join(lines) + '\n')

if __name__ == '__main__':
    arffPath, kVal = parseArgs(sys.argv)
    features, labels = getArffData(arffPath)
    classes = getUniqueClasses(labels)
    runEnsemble(features, labels, kVal)
    combinePredictions()
    writeOutput(arffPath, kVal, classes)
    print(f'results written to: {outputPath}')
