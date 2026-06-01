import math
import sys

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
            if line.startswith('@data'):
                break
            elif line.startswith('@attribute'):
                feature = line.split(' ')[1]
                features.append(feature)
            elif line.startswith('@data'):
                break
    return features,labels

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
        try:
            pVal = float(input("Enter a value for p: "))
        except ValueError:
            print("Invalid number! Using p = 3.")
            pVal = 3.0
    result = getDistance([testPoint, trainPoint], choice, pVal)
    print(f"distance: {getDistance([testPoint, trainPoint], choice, pVal)}")