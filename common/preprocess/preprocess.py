"""Preprocessing.

Final outputs:
    tempAnnotations
        -allImages.txt: file with the path of all images of interest
        -annotations.txt: file with annotations for each image in the following
        format:
            path class classIndex numberBoundingBoxes x1 y1 w1 h1 x2 y2 w2 h2 .
        -classes.txt: file with all of the class names found in the annotations

    textAnnotations: each image file has a corresponding txt file with its
    annotations
        -dayTrain--00001.txt
        -dayTrain--00002.txt
        -dayTrain--00003.txt

        annotations come in the following format:
            class, x y w h

    annotations.pkl: pickle file with all annotations in an organized
    dictionary. It has the following structure:
        Dict[imageFilePath]:
            Dict[classes]:
                List[All annotations of this class in the image]:
                    List[x, y, w, h]
"""

import os
import pickle
from typing import Dict, List, TextIO

colors = ["Green", "Yellow", "Red", "GreenLeft", "YellowLeft", "RedLeft"]


def processFromDir(dataDir: str, typing: str, dataFile: TextIO) -> None:
    """Process Color.

    Initially take the annotations from the original files for each of the
    given colors. The colors (in the form of typing) can be any of the above.

    Datadir - directory where the data file is
    typing - color that you are processing for
    dataFile - final file that will contain all of the annotations
    """
    dirs = list(filter(lambda x: os.path.isdir(x), os.listdir(dataDir)))

    for d in dirs:
        annoDir = "orginalAnnotations/trafficLightBox"
        path = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(path)

        processData: Dict[int, str] = {}

        # processes and saves all annotation files
        for f in dataFiles:
            if (typing + ".txt").lower() in f.lower():
                processData = process(open("{}/{}".format(path, f)).read(), d)

        annoDir = "frames"
        mainPath = "{}/{}/{}".format(dataDir, d, annoDir)
        files = os.listdir(mainPath)

        # Checks to see if file exists and then classifies it as positive or
        # negative data
        for f in files:
            f = f.replace("\n", "")

            # is picture
            if ((".png" not in f) and (".jpg" not in f)):
                continue

            index = int(f.split("--")[1].split(".")[0])
            val = processData.get(index, None)
            print(f)
            filePath = "{}/{}".format(mainPath, f)

            if val is not None:
                dataFile.write("{} {} {} {}\n".format(
                    os.path.abspath(filePath), typing, colors.index(
                        typing
                    ), " ".join(val.split(" ")[1:])))


def process(fileText: str, d: str) -> Dict[int, str]:
    """Create a data dictionary from a file.

    This function will take in an original annotations file and will return
    a dictionary with the given annotations

    fileText - text from the original csv file
    d - directory where this annotation was taken from for naming purposes
    """
    lineArray = fileText.split("\n")
    data = {}

    for i in range(len(lineArray)):
        line = lineArray[i].replace("\t", " ").replace("  ", " ").split(" ")

        if line[0] != "":
            index = int(line[0])
            textIndex = "0" * (5 - len(str(index))) + str(str(index))
            data[index] = "{}/frames/{}--{}.png {}".format(
                d, d, textIndex, " ".join(line[1:]))

    return data


def colorCycle(dataDir: str, outputDir: str) -> None:
    """Goes through all colors in traffic lights.

    This function simply cycles through the different colors and collects the
    original annotations

    dataDir - directory that contains the original data
    outputDir - directory where you want the annotations to be
    """
    dataFile = open(os.path.join(outputDir, "annotations.txt"), 'w')
    classesFile = open(os.path.join(outputDir, "classes.txt"), 'w')

    for col in colors:
        classesFile.write("{} {}\n".format(col, colors.index(col)))
        processFromDir(dataDir, col, dataFile)

    dataFile.close()
    classesFile.close()


def getAllImages(dataDir: str, outputDir: str) -> None:
    """Create a file with all images.

    This will create a file that contains the path to all of the images

    dataDir - directory that contains the original data
    outputDir - directory where you want the annotations to be
    """
    dirs = list(filter(lambda x: os.path.isdir(x), os.listdir(dataDir)))
    dataFile = open(os.path.join(outputDir, "allImages.txt"), 'w')

    for d in dirs:
        path = os.path.abspath(os.path.join(dataDir, d, "frames"))

        imgs = os.listdir(path)

        for img in imgs:
            dataFile.write("{}\n".format(os.path.join(path, img)))

    dataFile.close()


def setupDirs(outputDir: str) -> None:
    """Set up directories.

    Creates the directories necessary for the Annotations. Deletes earlier
    dataFiles if they exist in the directory

    outputDir - directory where you want the directories to be created in
    """
    try:
        os.mkdir(os.path.join(outputDir, "tempAnnotations"))
    except FileExistsError:
        for files in os.listdir(os.path.join(outputDir, "tempAnnotations")):
            os.remove(os.path.join(outputDir, "tempAnnotations", files))

    try:
        os.mkdir(os.path.join(outputDir, "textAnnotations"))
    except FileExistsError:
        for files in os.listdir(os.path.join(outputDir, "textAnnotations")):
            os.remove(os.path.join(outputDir, "textAnnotations", files))


def combineAnnotations(tempDir: str, outputDir: str) -> None:
    """Combine the annotations into a few select files.

    This function will use the intermediary annotations to create firstly a
    pickle file that contains all the annotations information in a dictionary.
    It also creates a new directory where there is a text file for each image
    file that contains the annotations for the corresponding image

    tempDir - directory where the temporary annotations are
    outputDir - directory where you want the final annotations to be
    """
    print("combining annotations...")
    dataDict: Dict[str, Dict[str, List[List[str]]]] = {}

    f = open(os.path.join(tempDir, "allImages.txt"), "r").read().split("\n")

    for lineText in f:
        if lineText == "":
            continue

        dataDict[lineText] = {}

    f = open(os.path.join(tempDir, "annotations.txt"), "r").read().split("\n")

    for lineText in f:
        if lineText == "":
            continue

        line = lineText.split(' ')
        classification = line[1]
        num = int(line[3])

        for i in range(num):
            dataVal: List[List[str]] = dataDict[line[0]].get(classification,
                                                             [])
            dataVal.append(line[4 + 4 * i:8 + 4 * i])
            dataDict[line[0]][classification] = dataVal

    p = open("annotations.pkl", 'wb')
    pickle.dump(dataDict, p, protocol=2)

    for image in dataDict.keys():
        newPath = "{}.txt".format(image.split("/")[-1].split(".")[0])
        n = open(os.path.join(outputDir, "textAnnotations", newPath), 'w')
        for classification in dataDict[image].keys():
            n.write("{}".format(classification))
            for data in dataDict[image][classification]:
                n.write(",{}".format(" ".join(data)))

            n.write("\n")
        n.close()


if __name__ == "__main__":
    setupDirs(".")
    colorCycle("../data/dayTrain", "tempAnnotations")
    getAllImages("../data/dayTrain", "tempAnnotations")
    combineAnnotations("tempAnnotations", ".")
