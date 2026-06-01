import math
import sys
import time

def getDistance(pointData, distMethod, pVal):
    total = 0
    if distMethod == '1': p = 2     # euclidean (power of 2)
    elif distMethod == '2': p = 1   # manhattan (power of 1)
    else: p = pVal                     # minkowski (custom pwr)

    for i in range(len(pointData[0])):
        total += abs(pointData[0][i] - pointData[1][i]) ** p
        
    return total ** (1 / p)

def getArffData(filename):
    features,labels = [],[]
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('@data') or line.startswith('@data'): break
            elif line.startswith('@attribute'): features.append(line.split(' ')[1])
    return features,labels

features,labels = getArffData(sys.argv[0])
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
    print(f"distance: {getDistance([testPoint, trainPoint], choice, pVal)}")

    endTime = time.time()
    print(f"operation time: {endTime - startTime} seconds")