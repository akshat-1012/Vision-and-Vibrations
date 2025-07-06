import numpy as np
from matplotlib import pyplot as plt

#We only plot every 8th row and 8th column just to reduce the arrow density
#And have a better looking plot

def get_avgmagnitude(u, v):
    scale = 2  
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg

def draw_quiver(u,v,beforeImg, afterImg):
    scale = 2


    fig, axs = plt.subplots(2,2,figsize=(beforeImg.shape[1] / 100, beforeImg.shape[0] / 100))
    axs[0, 0].imshow(beforeImg, cmap='gray')
    axs[0, 0].set_title("Input Image 1")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(afterImg, cmap='gray')
    axs[0, 1].set_title("Input Image 2")
    axs[0, 1].axis('off')
    ax = axs[1, 0]
    ax.imshow(beforeImg, cmap = 'gray')
    magnitudeAvg = get_avgmagnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            
            #plot only the significant arrows
            if magnitude > magnitudeAvg:
                ax.quiver(j,i, dx, dy, color = 'red',angles='xy', scale_units='xy', scale=1,  width=0.002)
    ax.axis('off')
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

