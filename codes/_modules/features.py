"""
This module is for feature extraction for an image in OpenCV format.
"""

import numpy as np
import math


def frameToBlocks(rawFrame: list, rows: int, cols: int):
    """
    This function is for dividing a frame into blocks for each color channels.
    ================================================================
    Input:
      rawFrame: list[height][width][channel] = intensity
    Output: (for example)
      frameToBlocks(rawFrame, 2, 3) = blocks[ ] with length 24
    where
      blocks[0] = list of intensity for pixels in 1st block BLUE channel
    and
      blocks[1] = list of intensity for pixels in 1st block GREEN channel
    and
      blocks[2] = list of intensity for pixels in 1st block RED channel
    """
    rawH = len(rawFrame)          # Heigth of Frame
    rawW = len(rawFrame[0])       # Width of Frame
    rawC = len(rawFrame[0][0])    # Nubmer of channel of Frame
    blockH = rawH // rows         # Heigth of Block
    blockW = rawW // cols         # Heigth of Block
    blockNum = rows * cols * 3    # Number of Blocks for a Frame

    imageBlocks = []              # Declare size of imageBlocks
    for p in range(0, blockNum):
        imageBlocks.append([])

    for i in range(0, rawH):      # For each pixel, append into a block
        for j in range(0, rawW):  # decide by spatial coord. of that pixel
            for k in range(0, rawC):
                blockKey = (
                    min((i // blockH), rows - 1) * rawC +
                    min((j // blockW), cols - 1) * rawC * rows +
                    k
                )
                imageBlocks[blockKey].append(rawFrame[i][j][k])
    # Block Number (e.g row 2 * col 4 * channel 3):
    # (0,0,0) (0,1,0) (0,2,0) (0,3,0)        0  6 12 18
    # (0,0,1) (0,1,1) (0,2,1) (0,3,1)        1  7 13 19
    # (0,0,2) (0,1,2) (0,2,2) (0,3,2)        2  8 14 20
    # (1,0,0) (1,1,0) (1,2,0) (1,3,0) --->   3  9 15 21
    # (1,0,1) (1,1,1) (1,2,1) (1,3,1)        4 10 16 22
    # (1,0,2) (1,1,2) (1,2,2) (1,3,2)        5 11 17 23
    return imageBlocks


def blocksToFeatures(rawBlocks: list):
    """
    This function is for transforming blocks to feature vectors.
    ================================================================
    {Simplified Color-based}
      For eack block, calculate the average of intensity
    {Simplified Principal Component Analysis }
      Then reduce the dimension by discret consine transform.
      Truncate the transformed feature vector.
      From (Row*Col*ColorChannel)-demensions to 16-demensions.
    """
    tempFeaures = []
    blocksFeatures = []
    # Calculate average intensity for each block
    for p in range(0, len(rawBlocks)):
        tempFeaures.append(sum(rawBlocks[p]) // len(rawBlocks[p]))
    # Run discrete consine transfrom on color information
        # TBC
        blocksFeatures = _discretConsineTransform(tempFeaures, 16)
    return blocksFeatures


def _discretConsineTransform(rawVector: list, dimensionRetained: int):
    """
    This function is for transforming a high-dimension vector by DCT.
    ================================================================
    The transformed vector is truncated to a low-dimension vector.
    """

    # No transform if original dimension = 1
    if (len(rawVector) == 1):
        return rawVector

    # Create Matrix for DCT
    dimensionOriginal = len(rawVector)
    transformVectors = [[]]
    resultVector = []
    normalizedCoeff = math.sqrt(2/dimensionOriginal)

    # For u = 0
    for q in range(0, dimensionOriginal):
        transformVectors[0].append(1 / math.sqrt(dimensionOriginal))
    resultVector.append(int(np.dot(transformVectors[0], rawVector)))
    # For u > 0
    for p in range(1, dimensionRetained):
        transformVectors.append([])
        for q in range(0, dimensionOriginal):
            if (p >= dimensionOriginal):
                transformVectors[p].append(0)
            else:
                transformVectors[p].append(
                    normalizedCoeff *
                    math.cos(
                        ((2 * q + 1) * p * math.pi) /
                        (2 * dimensionOriginal)))
        resultVector.append(int(np.dot(transformVectors[p], rawVector)))
    return resultVector
