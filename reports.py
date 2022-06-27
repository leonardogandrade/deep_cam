import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import json
import os
from sys import path
import numpy as np

class Reports:
    def __init__(self):
        self.outputPath = os.path.join(os.getcwd(), "output")
        self.colors = ["#1D5C96", "#6497B2", "#B3CEE1", "#07214C", "#10396D"]
        self.imagePath = os.path.join(os.getcwd(), "results")

    def loadFile(self, filename):
        file = open(os.path.join(self.outputPath, filename), 'r')
        data = json.load(file)
        file.close()
        return data

    def lineChart(self):
        plt.plot([1,2,3],[10,20,30])
        plt.show()

    def heatMap(self,data):
        y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

        z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, remove the last value from the z array.
        z = z[:-1, :-1]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        fig, ax = plt.subplots()

        c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        # ax.set_title('pcolormesh')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)
        plt.axis('off')
        plt.savefig(os.path.join(self.imagePath, "heatmap.png"))

    def pie(self, data):
        colors = self.colors
        labels = []
        info = []

        for i in data["fences"]:
            labels.append(i["fenceId"])
            info.append(i["totalTimeFence"])

        labels.append("total video")
        info.append(data["video"]["totalTime"])
        
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(info, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90, colors=colors)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig(os.path.join(self.imagePath, "fenceTime.png"))

    def bar(self, data):
        classes = list(data["expressions"].keys())
        values = list(data["expressions"].values())
        fig = plt.figure(figsize = (10, 5))
        plt.bar(classes, values, width = 0.4, color=self.colors)
        
        plt.xlabel("Tipos de expressoes")
        plt.ylabel("Expressoes por frame")
        plt.title("Expressoes Faciais")
        plt.savefig(os.path.join(self.imagePath, "expressions.png"))

    def output(self, payload, outputName):
        filename = os.path.join(self.outputPath, outputName + ".json")
        
        if os.path.exists(filename):
            os.remove(filename)

        file = open(filename, "w")
        json.dump(payload, file, indent=5)
        file.close()
        
    def allInOne(self):
        cm = 1/2.54
        fig, axis = plt.subplots(2, 2, figsize=(20*cm, 10*cm))

        expressions = plt.imread(self.imagePath + "/expressions.png")
        ageGender = plt.imread(self.imagePath + "/ageGender.png")
        fenceTime = plt.imread(self.imagePath + "/fenceTime.png")
        heatmap = plt.imread(self.imagePath + "/heatmap.png")
        
        axis[0, 0].imshow(fenceTime)
        axis[0, 0].set_title('Permanencia')
        axis[0, 0].axis('off')
        axis[0, 1].imshow(expressions)
        axis[0, 1].set_title('Expressoes')
        axis[0, 1].axis('off')
        axis[1, 0].imshow(heatmap)
        axis[1, 0].set_title('Heatmap')
        axis[1, 0].axis('off')
        axis[1, 1].imshow(ageGender)
        axis[1, 1].set_title('Gênero e idade')
        axis[1, 1].axis('off')

        fig.tight_layout()
        fig.canvas.set_window_title('Relatorio')
        
        plt.show()


reports = Reports()

# reports.output(p, "fence-output")
fenceDataFile = reports.loadFile("fence-output.json")
expressionsDataFile = reports.loadFile("expressions-output.json")
#reports.heatMap(data)
#reports.scatter(data)

#reports.pie(fenceDataFile)
# expressionsDataFile = reports.loadFile("expressions-output.json")
# reports.bar(expressionsDataFile)

reports.pie(fenceDataFile)
reports.heatMap(fenceDataFile)
reports.bar(expressionsDataFile)
reports.allInOne()

