"""
This module is for searching among multiple-dimension numerical index.
For simplicity, Octree would be builded for searching.
Only the top-3 features are used for indexing.
"""

import time

##############################################################################
# ----------Functions for external scripts: indexing() and searching()--------
##############################################################################


def indexing(featureVectors: list, maxChild: int):
    """
    Building index tree of feature vectors.
    ================================================================
    Output Format:  Linked list of [[header], childrens]
        where [Header] = [Leaf: Bool, xLow, yLow, zLow, xHigh, yHigh, zHigh, maxChild]
    For Leaf = False,
        maxChild = 8 AND
        childrens = sub-index
    For Leaf = True,
        maxChild is parameter AND
        childrens = feature vectors in [time, [features]]
    """

    # Building root node
    tree = []
    tree.append([True] + _intialBroundary(featureVectors) + [maxChild])

    # Start insertion
    start = time.time()
    for i in range(0, len(featureVectors)):

        _treeInsertion(tree, featureVectors[i])

        # Show progress in percentage
        now = time.time()
        if ((now - start) > 6):
            progressPercentage = int(i / len(featureVectors) * 100)
            print(f'> Progress: {progressPercentage}%')
            start = time.time()

    return tree


def searching(index: list, features: list, counter=0):
    """
    Searching in index tree of feature vectors.
    ================================================================
    Output Format: (exactMatch, similarMatch, counter) where
        exactMatch: list of feature vectors in format [time, [features]]
        similarMatch: list of feature vectors in format [time, [features]]
        counter: number of comparison
    """
    if (not index[0][0]):
        # Search in intermediate node
        # In worst case, 8 comparison are needed.
        location = 0
        locationFlag = False
        while (not locationFlag):
            location += 1
            locationFlag = _compareBroundary(
                index[location][0][1:7], features[0:3])
            counter += 1
        return searching(index[location], features, counter)
    else:
        # Search in leaf node
        # In worst case, maxChild of comparisons are needed.
        exactMatch = []
        similarMatch = []
        for i in range(1, len(index)):
            counter += 1
            if (_distanceLowerBound(index[i][1], features) < 32):   # arbitrarily number
                similarMatch.append(index[i])                       # not optimized
                if (_distanceActual(index[i][1], features) < 1):
                    exactMatch.append(index[i])
        return exactMatch, similarMatch, counter


##############################################################################
# ----------Functions for internal uses: Determine/ Compare Coordination------
##############################################################################

def _intialBroundary(featureVectors: list):
    # Assume the top-3 features are [x, y, z]
    # Intial broundary in [Min(x), Min(y), Min(x), Max(x), Max(y), Max(x)] among all features
    boundary = [
        featureVectors[0][1][0],
        featureVectors[0][1][1],
        featureVectors[0][1][2],
        featureVectors[0][1][0],
        featureVectors[0][1][1],
        featureVectors[0][1][2]
    ]
    for i in range(1, len(featureVectors)):
        # Find Min(x) and Max(x)
        if (featureVectors[i][1][0] < boundary[0]):
            boundary[0] = featureVectors[i][1][0]
        elif (featureVectors[i][1][0] > boundary[3]):
            boundary[3] = featureVectors[i][1][0]
        # Find Min(y) and Max(y))
        if (featureVectors[i][1][1] < boundary[1]):
            boundary[1] = featureVectors[i][1][1]
        elif (featureVectors[i][1][1] > boundary[4]):
            boundary[4] = featureVectors[i][1][1]
        # Find Min(z) and Max(z)
        if (featureVectors[i][1][2] < boundary[2]):
            boundary[2] = featureVectors[i][1][2]
        elif (featureVectors[i][1][2] > boundary[5]):
            boundary[5] = featureVectors[i][1][2]
    return boundary


def _subBroundarys(inputBroundary: list):
    # Assume input broundary is [xLow, yLow, zLow, xHigh, yHigh, zHigh].
    # It will be divided into 8 cubes:
    #   000:    [xLow, yLox, zLow, xMid, yMid, zMid]
    #   001:    [xLow, yLox, zMid, xMid, yMid, zHigh]
    #   010:    [xLow, yMid, zLow, xMid, yHigh, zMid]
    #   011:    [xLow, yMid, zMid, xMid, yHigh, zHigh]
    #   100:    [xMid, yLox, zLow, xHigh, yMid, zMid]
    #   101:    [xMid, yLox, zMid, xHigh, yMid, zHigh]
    #   110:    [xMid, yMid, zLow, xHigh, yHigh, zMid]
    #   111:    [xMid, yMid, zMid, xHigh, yHigh, zHigh]
    xMid = (inputBroundary[0] + inputBroundary[3]) / 2
    yMid = (inputBroundary[1] + inputBroundary[4]) / 2
    zMid = (inputBroundary[2] + inputBroundary[5]) / 2
    outputBoundarys = []
    # Create 000
    temp = inputBroundary.copy()
    temp[3], temp[4], temp[5] = xMid, yMid, zMid
    outputBoundarys.append(temp)
    # Create 001
    temp = inputBroundary.copy()
    temp[3], temp[4], temp[2] = xMid, yMid, zMid
    outputBoundarys.append(temp)
    # Create 010
    temp = inputBroundary.copy()
    temp[3], temp[1], temp[5] = xMid, yMid, zMid
    outputBoundarys.append(temp)
    # Create 011
    temp = inputBroundary.copy()
    temp[3], temp[1], temp[2] = xMid, yMid, zMid
    outputBoundarys.append(temp)
    # Create 100
    temp = inputBroundary.copy()
    temp[0], temp[4], temp[5] = xMid, yMid, zMid
    outputBoundarys.append(temp)
    # Create 101
    temp = inputBroundary.copy()
    temp[0], temp[4], temp[2] = xMid, yMid, zMid
    outputBoundarys.append(temp)
    # Create 110
    temp = inputBroundary.copy()
    temp[0], temp[1], temp[5] = xMid, yMid, zMid
    outputBoundarys.append(temp)
    # Create 111
    temp = inputBroundary.copy()
    temp[0], temp[1], temp[2] = xMid, yMid, zMid
    outputBoundarys.append(temp)

    return outputBoundarys


def _compareBroundary(inputBroundary: list, coordination: list):
    # Return true if a coordination is within broader
    if (
        coordination[0] >= inputBroundary[0] and
        coordination[0] <= inputBroundary[3]
    ):
        if (
            coordination[1] >= inputBroundary[1] and
            coordination[1] <= inputBroundary[4]
        ):
            if (
                coordination[2] >= inputBroundary[2] and
                coordination[2] <= inputBroundary[5]
            ):
                return True
    return False

##############################################################################
# ----------Functions for internal uses: Building Index Tree----------
##############################################################################


def _treeInsertion(inputTree: list, newleaf: list):
    # Recusive function to insert leafs at bottom.

    # Case 1:   If root of inputTree is not full, then directly insert.
    if (inputTree[0][0]):
        inputTree.append(newleaf)
        # Check if root of inputTree is full, split it.
        # inputTree could be a subtree for this recusive function.
        if (len(inputTree) > inputTree[0][7]):
            _treeSplit(inputTree)
        return inputTree

    # Case 2:   If root of inputTree is full, then insert in subtree.
    else:
        # Compare to determine which subtree be inserted to.
        # It is a Octree. Each intermediate node has 8 sub-node.
        location = 0
        locationFlag = False
        while (not locationFlag):
            location += 1
            locationFlag = _compareBroundary(
                inputTree[location][0][1:7], newleaf[1][0:3])
        _treeInsertion(inputTree[location], newleaf)
        return inputTree


def _treeSplit(inputTree: list):
    # Tree is splited if it is full. Leaf nodes will be assigned to sub-trees.

    # Step 1: Store children of inputTree in temparery list
    tempList = []
    for i in range(1, inputTree[0][7] + 1):
        tempList.append(inputTree.pop())

    # Step 2: Replace inputTree root(=leaf) by a splited node (=intermediate node)
    #   Insert intermediate nodes
    tempCoord = _subBroundarys(inputTree[0][1:7])
    for i in range(0, 8):
        inputTree.append([[True] + tempCoord[i] + [inputTree[0][7]]])
    #   Modify header information
    inputTree[0][0] = False
    inputTree[0][7] = 8

    # Step 3: Assign children back to splited tree
    for i in range(0, len(tempList)):
        _treeInsertion(inputTree, tempList[i])

    return inputTree

##############################################################################
# ----------Functions for internal uses: Search in Index Tree----------
##############################################################################


def _distanceActual(vector1: list, vector2: list):
    output = 0
    for i in range(0, len(vector1)):
        output += (vector1[i] - vector2[i]) ** 2
    return (output ** 0.5) / len(vector1)


def _distanceLowerBound(vector1: list, vector2: list):
    output = 0
    for i in range(0, 3):
        output += (vector1[i] - vector2[i]) ** 2
    return int((output ** 0.5) // 3)

