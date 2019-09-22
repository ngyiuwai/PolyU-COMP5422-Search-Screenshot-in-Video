import os
import json
import cv2
from _modules import features
from _modules import searchtree

##############################################################################
# ----------STEP 1: Import feature vectors in JSON----------
##############################################################################
print('> Please move the JSON feature vectors into folder "output".')
featuresName = input('> Please enter filename here, e.g. "sample.mp4.2x8.json": ')

print('> ...Reading JSON feature vectors...')
featuresName = 'output/' + featuresName
script_dir = os.path.dirname(__file__)
relative_path = featuresName
abs_file_path = os.path.join(script_dir, relative_path)
featuresFile = open(abs_file_path, "r")
if (featuresFile is None):
    print('> ERROR is reading JSON feature vectors.')
else:
    print('> Done. Search tree would be builded using the feature vectors.')
featuresList = json.loads(featuresFile.read())
print('')

##############################################################################
# ----------STEP 2: Build a tree using feature vectors for searching----------
##############################################################################
print('> ...Building search tree using feature vectors...')
tree = searchtree.indexing(featuresList, int((len(featuresList) ** 0.5)))

treeName = 'output/index_' + featuresName[7:]
print('> ...Saving index into JSON file...')
save_path = os.path.join(script_dir, treeName)
textfile = open(save_path, "w")
textfile.write(json.dumps(tree))
textfile.close()
print('> Done. Search tree is saved in JSON under folder "output"')
print('')

##############################################################################
# ----------STEP 3: Import a screenshot and convert to features----------
##############################################################################
# Read screenshot
print('> Please move a screenshoot image into folder "input".')
targetName = input('> Please enter filename here, e.g. "screenshot.jpg": ')

print('> ...Reading image...')
targetName = 'input/' + targetName
script_dir = os.path.dirname(__file__)
relative_path = targetName
abs_file_path = os.path.join(script_dir, relative_path)
targetImg = cv2.imread(abs_file_path)
if (targetImg is None):
    print('> ERROR is reading image')
else:
    print('> Done. Image would be divided into blocks for feature extraction.')

nRow = int(input('> Please input the number of row: '))
nCol = int(input('> Please input the number of column: '))
print('> ...Dividing image to blocks and extracting features...')
targetFeature = features.blocksToFeatures(
    features.frameToBlocks(targetImg, nRow, nCol)
)
print('> Done. Corresponding features cector:')
print('> ', targetFeature)
print('')

##############################################################################
# ----------STEP 4: Search features among the tree and return result----------
##############################################################################

print('> ...Comparing feature vectors')
searchResult = searchtree.searching(tree, targetFeature)
print(f'> Done. {searchResult[2]} comparisons are performed among {len(featuresList)} features vectors.')
print('> Exact match results: ', targetFeature)
for i in range(0, len(searchResult[0])):
    print('>>> ', searchResult[0][i])
print('> Similar match results: ', targetFeature)
for i in range(0, len(searchResult[1])):
    print('>>> ', searchResult[1][i])
print('')

input('> ...Please press Enter to exit...')
