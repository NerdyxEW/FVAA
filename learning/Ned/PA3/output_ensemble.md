# kNN Ensemble Report

arff file: learning/resources/data/small.arff
k: 3

## Individual Classifiers

### euclidean - distance 1

```
                    0       1       2       3       4       5       6       7
0                 137       0       0       0       0       0       0       6
1                   5      57       0       0      14       0       0       1
2                   0       0       0       0       1       0       0       1
3                   0       1       0       0       0       0       1       0
4                   1      13       1       0      19       0       0       1
5                   1       1       0       0       0      14       0       4
6                   0       0       0       0       0       0       5       0
7                   7       2       1       0       2       2       0      38
```

accuracy: 270/336 (0.8036)

### manhattan - distance 2

```
                    0       1       2       3       4       5       6       7
0                 138       1       0       0       0       0       0       4
1                   4      53       0       1      16       0       0       3
2                   0       0       0       0       1       0       0       1
3                   0       1       0       0       0       0       1       0
4                   1      14       1       0      19       0       0       0
5                   1       1       0       0       0      13       0       5
6                   0       0       0       0       0       0       5       0
7                   9       1       1       0       2       2       0      37
```

accuracy: 265/336 (0.7887)

### minkowski - distance 3 (p=3.0)

```
                    0       1       2       3       4       5       6       7
0                 137       1       0       0       0       0       0       5
1                   5      56       0       0      15       0       0       1
2                   0       0       0       0       1       0       0       1
3                   0       1       0       0       0       0       1       0
4                   1      13       0       0      19       0       1       1
5                   1       0       0       0       1      13       0       5
6                   0       0       0       0       0       0       5       0
7                   6       2       1       0       2       1       0      40
```

accuracy: 270/336 (0.8036)

## Ensemble

combination: majority vote across euclidean, manhattan, and minkowski classifiers

```
                    0       1       2       3       4       5       6       7
0                 137       1       0       0       0       0       0       5
1                   5      57       0       0      14       0       0       1
2                   0       0       0       0       1       0       0       1
3                   0       1       0       0       0       0       1       0
4                   1      13       1       0      19       0       0       1
5                   1       1       0       0       0      13       0       5
6                   0       0       0       0       0       0       5       0
7                   7       2       1       0       2       2       0      38
```

accuracy: 269/336 (0.8006)
