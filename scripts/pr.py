import numpy as np
import matplotlib.pyplot as plt

output = np.array(np.load("output.npy"))

print(output)

green = output[:, 0]
red = output[:, 4]
off = output[:, 5]
yellow = output[:, 6]

plt.title("P/R Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.plot(green[:, 1], green[:, 0], "xkcd:green", label="green")
plt.plot(red[:, 1], red[:, 0], "xkcd:red", label="red")
plt.plot(off[:, 1], off[:, 0], "xkcd:black", label="off")
plt.plot(yellow[:, 1], yellow[:, 0], "xkcd:yellow", label="yellow")

plt.legend()
plt.grid()
plt.savefig("pr.png")
plt.savefig("pr.pdf")
