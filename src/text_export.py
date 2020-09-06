import os.path


class TextExport:

    number = 1

    def __init__(self, pathAndFileName):
        if os.path.isfile(pathAndFileName):
            self.file = open(pathAndFileName, "a+");
        else:
            self.file = open(pathAndFileName, "a+");
            self.file.write("Found Expressions:\n");

    def append(self, frameNumber, px, py, expression):
        self.file.write("  Number: {}\n".format(self.number))
        self.file.write("    Frame: {} \n".format(frameNumber))
        self.file.write("    Position: {} {} \n".format(str(px), str(py)))
        self.file.write("    Expression: {} \n".format(expression))
        self.number += 1

    def close(self):
        self.file.close()

if __name__ == "__main__":
    export = TextExport("test.txt")
    export.append(3,(11,22),(33,44),"happy")