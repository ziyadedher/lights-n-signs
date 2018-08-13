import os
import cv2

def processFromDir(dataDir, outputDir, typing):
    dirs = os.listdir(dataDir)
    neg = open("{}/{}_neg.txt".format(outputDir, typing), 'w')
    pos = open("{}/{}_pos.txt".format(outputDir, typing), "w")

    #Gets rid of all non directories
    deleted = 0
    for i in range(len(dirs)):
        if "." in dirs[i - deleted]:
            del dirs[i-deleted]
            deleted += 1

    for d in dirs:
        annoDir = "orginalAnnotations/trafficLightBox"
        path = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(path)

        processData = {}

        print("Processing annotations in directory {} for {}".format(d, typing))

        #processes and saves all annotation files
        for f in dataFiles:
            print(f)
            if (typing + ".txt").lower() in f.lower():
                print("This file meets typing specifications")
                processData = process(open("{}/{}".format(path, f)).read(), d)

        annoDir = "frames"
        mainPath = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(mainPath)

        #Checks to see if file exists and then classifies it as positive or negative data
        for f in dataFiles:
            f = f.replace("\n", "")

            #is picture
            if ((".png" not in f) and (".jpg" not in f)):
                continue

            index = int(f.split("--")[1].split(".")[0])
            val = processData.get(index, -1)
            print(f)
            filePath = "{}/{}".format(mainPath, f)

            if val == -1:
                neg.write("{}\n".format(filePath))
            else:
                pos.write("{} {}\n".format(filePath, " ".join(val.split(" ")[1:])))

    neg.close()
    pos.close()


def process(fileText, d):
    fileText = fileText.split("\n")
    data = {}

    for i in range(len(fileText)):
        line = fileText[i].replace("\t"," ").replace("  "," ").split(" ")

        if line[0] != "":
            index = int(line[0])
            textIndex = "0"*(5-len(str(index))) + str(str(index))
            num = int(line[1])
            data[index] = "{}/frames/{}--{}.png {}".format(d, d, textIndex, " ".join(line[1:]))

    return data

def greyImages(dataDir, outputDir):
    #replaces all images with their greyscale histogram equalized version
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    dirs = os.listdir(dataDir)

    for d in dirs:
        if "." in d:
            continue
        annoDir = "frames"
        mainPath = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(mainPath)

        for f in dataFiles:
            if ((".png" not in f) and (".jpg" not in f)):
                continue

            filePath = "{}/{}".format(mainPath, f)
            print(filePath)
            img = cv2.imread(filePath, 0)
            gray = clahe.apply(img)
            cv2.imwrite(filePath, gray)

#Creates a greyscale image out of the red pixels of the image if you want to isolate
#a specific light color
def greyRedImages(dataDir, outputDir):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    dirs = os.listdir(dataDir)

    for d in dirs:
        if "." in d:
            continue
        annoDir = "frames"
        mainPath = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(mainPath)

        for f in dataFiles:
            if ((".png" not in f) and (".jpg" not in f)):
                continue

            filePath = "{}/{}".format(mainPath, f)
            print(filePath)
            img = cv2.imread(filePath, 1)
            img = img[:,:,0]
            gray = clahe.apply(img)
            cv2.imwrite(filePath, gray)

#Creates a greyscale image out of the blue pixels of the image if you want to isolate
#a specific light color
def greyBlueImages(dataDir, outputDir):
    #replaces all images with their greyscale histogram equalized version
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    dirs = os.listdir(dataDir)

    for d in dirs:
        if "." in d:
            continue
        annoDir = "frames"
        mainPath = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(mainPath)

        for f in dataFiles:
            if ((".png" not in f) and (".jpg" not in f)):
                continue

            filePath = "{}/{}".format(mainPath, f)
            print(filePath)
            img = cv2.imread(filePath, 1)
            img = img[:,:,2]
            gray = clahe.apply(img)
            cv2.imwrite(filePath, gray)

#Creates a greyscale image out of the green pixels of the image if you want to isolate
#a specific light color
def greyGreenImages(dataDir, outputDir):
    #replaces all images with their greyscale histogram equalized version
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    dirs = os.listdir(dataDir)

    for d in dirs:
        if "." in d:
            continue
        annoDir = "frames"
        mainPath = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(mainPath)

        for f in dataFiles:
            if ((".png" not in f) and (".jpg" not in f)):
                continue

            filePath = "{}/{}".format(mainPath, f)
            print(filePath)
            img = cv2.imread(filePath,1)
            img = img[:,:,1]
            gray = clahe.apply(img)
            cv2.imwrite(filePath, gray)

#is called to cycle through all possible lights that could show up. Change to only process one kind of trafficLightBox
#lights
def colorCycle(dataDir, outputDir):
    colors = ["Green", "Yellow", "Red", "GreenLeft", "YellowLeft", "RedLeft"]

    for col in colors:
        processFromDir(dataDir, outputDir, col)

    #greyImages(dataDir, outputDir)
    #processYolo(dataDir, outputDir)


if __name__=="__main__":
    colorCycle("../data/dayTrain", "../haar")
