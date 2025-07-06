import numpy as np

def non_max_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q, r = 255, 255
                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = G[i, j + 1]
                    r = G[i, j - 1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = G[i + 1, j - 1]
                    r = G[i - 1, j + 1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = G[i + 1, j]
                    r = G[i - 1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = G[i - 1, j - 1]
                    r = G[i + 1, j + 1]

                if G[i, j] >= q and G[i, j] >= r:
                    Z[i, j] = G[i, j]
                else:
                    Z[i, j] = 0
            except IndexError as e:
                pass

    return Z
