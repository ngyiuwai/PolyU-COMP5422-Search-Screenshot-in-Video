import os
import json
import cv2
import time
from _modules import features

##############################################################################
# ----------STEP 1: Transform Video to Frames for further processing----------
##############################################################################


# Initialized the environment by creating paths
if not os.path.exists('input'):
    os.mkdir('input')
if not os.path.exists('output'):
    os.mkdir('output')

# Read video
print('> Please move a video into folder "input".')
userinputName = input('> Please enter filename here, e.g. "sample.mp4": ')
videoName = 'input/' + userinputName
script_dir = os.path.dirname(__file__)
relative_path = videoName
abs_file_path = os.path.join(script_dir, relative_path)
cap = cv2.VideoCapture(abs_file_path)
if (cap is None):
    print('> ERROR is reading JSON feature vectors.')

# Extract metadata of video: duration & numbmer of frames
print('> ...Reading video...')
fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frameCount/fps
print('> frame per second = ' + str(fps))
print('> number of frames = ' + str(frameCount))
print('> duration (in seconds) = ' + str(duration))

# Transform video to list of frames
# Each frame is a 3D array of uint8
# frames[frame#][height][width][channel] = intensity of one pixel
nextFrame = True
frames = []
start = time.time()
while(nextFrame):

    # Show progress in percentage
    now = time.time()
    if ((now - start) > 6):
        progressPercentage = int(len(frames) / frameCount * 100)
        print(f'> Progress: {progressPercentage}%')
        start = time.time()

    nextFrame, frame = cap.read()
    frames.append(frame)
cap.release()
print('> Done. Frames would be divided into blocks for feature extraction.')
print('')

##############################################################################
# ----------STEP 2: Transform Frames to Blocks and extract feature----------
##############################################################################

# Now start extracting feature vectors
# frameBlocks = list[block], where block is list of intensity in a block
# frameFeature = list[features for a frame], extracted from multiple blocks
# frameFeatures = list[frameFeature]
frameBlocks = []
frameFeature = []
frameFeatures = []
for p in range(0, frameCount):  # Declare size of frameFeatures
    frameFeatures.append([])
nRow = int(input('> Please input the number of row: '))
nCol = int(input('> Please input the number of column: '))
nSkip = int(
    input('> Please input how much frames to skip between each reading: '))

print('> ...Dividing frames to blocks and extracting features...')

start = time.time()
count = 0
while (count < frameCount):

    # Show progress in percentage
    now = time.time()
    if ((now - start) > 6):
        progressPercentage = int(count / frameCount * 100)
        print(f'> Progress: {progressPercentage}%')
        start = time.time()

    # Insert time into vector, which serves as filePath of an image
    timepoint = count/fps
    frameFeatures[count].append(timepoint)

    # Insert features into vector, which serves as UniqueID of an image
    frameBlocks = features.frameToBlocks(frames[count], nRow, nCol)
    frameFeature = features.blocksToFeatures(frameBlocks)
    frameFeatures[count].append(frameFeature)

    # Skip Frames to speed up
    count += nSkip + 1

print('> Done. Features Vectors would be saved in JSON.')
print('')

# Remove empty elements
stop = False
q = 0
while (not stop):
    if (frameFeatures[q] == []):
        frameFeatures.pop(q)
    else:
        q += 1
    if (q == len(frameFeatures)):
        stop = True

##############################################################################
# ----------STEP 3: Write feature on disk in JSON format----------
##############################################################################

outputName = userinputName + '.' + str(nRow) + 'x' + str(nCol) + '.json'
filename = 'output/' + outputName
print('> ...Saving feature vectors into JSON file...')
save_path = os.path.join(script_dir, filename)
textfile = open(save_path, "w")
textfile.write(json.dumps(frameFeatures))
textfile.close()
print('> Done. Features Vectors is saved in JSON under folder "output"')
print('> File name: ', outputName)
print('> You may search screenshots by features using "searcher.py".')
input('> ...Please press Enter to exit...')
