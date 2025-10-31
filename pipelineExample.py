import subprocess
import os
import cv2
import numpy as np


# Example funciton that generates random noise and add it to the image. Clamp values to [0, 255].
def add_gaussian_noise(image, mean=0, std=10): # Visually this is not very noticeable, but I am testing with COG which breaks quickly.
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return noisy_image

# Generate a star field image as a base.
subprocess.run(["./lost", "pipeline",
                "--generate", "1",
                "--generate-x-resolution", "1024",
                "--generate-y-resolution", "1024",
                "--fov", "30",
                #"--generate-reference-brightness", "100", -- Not functional for me? May need to rebuild.
                "--generate-spread-stddev", "1",
                "--generate-read-noise-stddev", "0.05",
                "--generate-ra", "88",
                "--generate-de", "7",
                "--generate-roll", "0",
                "--plot-raw-input", "raw-input.png",
                "--plot-input", "input.png",])



#------------------------------------------------
# OpenCV example:
image = cv2.imread('raw-input.png')

# Image exists, add noise
noisy_image = add_gaussian_noise(image)

# If we want to display them:
cv2.imshow("Original Image", image)
cv2.imshow("Noisy Image", noisy_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the noisy image.
cv2.imwrite('noisy_input.png', noisy_image) 


database_path = "test-database.dat"

if not os.path.exists(database_path):
    subprocess.run([
        "./lost", "database",
        "--max-stars", "5000",
        "--kvector",
        "--kvector-min-distance", "0.2",
        "--kvector-max-distance", "15",
        "--kvector-distance-bins", "10000",
        "--output", database_path
    ])

result = subprocess.run([
        "./lost", "pipeline",
        "--png", "noisy_input.png",
        "--focal-length", "30",
        "--pixel-size", "22.2",
        "--centroid-algo", "cog",
        "--centroid-mag-filter", "5",
        "--database", database_path,
        "--star-id-algo", "py",
        "--angular-tolerance", "0.05",
        "--false-stars", "1000",
        "--max-mismatch-prob", "0.0001",
        "--attitude-algo", "dqm",
        "--print-attitude", "attitudeNoise.txt"
    ], capture_output=True, text=True)

print("LOST Output:")
print(result)
print(result.stdout)

result = subprocess.run([
        "./lost", "pipeline",
        "--png", "raw-input.png",
        "--focal-length", "30",
        "--pixel-size", "22.2",
        "--centroid-algo", "cog",
        "--centroid-mag-filter", "5",
        "--database", database_path,
        "--star-id-algo", "py",
        "--angular-tolerance", "0.05",
        "--false-stars", "1000",
        "--max-mismatch-prob", "0.0001",
        "--attitude-algo", "dqm",
        "--print-attitude", "attitudeBase.txt"

    ], capture_output=True, text=True)

print("LOST Output:")
print(result)
print(result.stdout)


