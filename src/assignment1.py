import math
import sys
import time

def getDistance(pointData, distMethod, pVal):
    total = 0
    for i in range(len(pointData[0])):
        total += abs(pointData[0][i] - pointData[1][i]) ** pVal
    return total ** (1 / pVal)

def getArffData(filename):
    features,labels = [],[]
    data = False
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().lower()
            if not line or line.startswith('@'): continue
            if line.startswith('@data'): data = True continue
            elif line.startswith('@attribute'): continue
            if data:
                p = line.split(',')
                features.append([float(v) for v in p[:-1]])
                labels.append(p[-1].strip())
    return features,labels

if len(sys.argv) < 2: sys.exit(1)
features,labels = getArffData(sys.argv[1])

testPoint = [2.1, 3.2, 4.3, 5.4]
trainPoint = [4.2, 6.4, 8.6, 10.8]

while True:
    print("\nChoose a distance method:\n[1] Euclidean\n[2] Manhattan\n[3] Minkowski")
    choice = input("Enter your choice: ")
    if choice not in ['1', '2', '3']:
        print("Fake number, retry!\n")
        continue
    pVal = 2.0
    if choice == '3':
        try: pVal = float(input("Enter a value for p: "))
        except ValueError:
            print("Invalid number! Using p = 3.")
            pVal = 3.0
    
    try: kVal = int(input("Enter a value for k: ")) # unfinished
    except ValueError: kVal = 1

    matrix = {actual: {pred: 0 for pred in labels} for actual in labels} # unfinished
    
    startTime = time.time()
    print(f"distance: {getDistance([testPoint, trainPoint], choice, pVal if choice == '3' else int(choice))}")

    endTime = time.time()
    print(f"operation time: {endTime - startTime} seconds")