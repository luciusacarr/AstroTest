import subprocess
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Used in Pillow example
starting_roll = 0


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
                "--generate-roll", str(starting_roll),
                "--plot-raw-input", "raw-input.png",
                "--plot-input", "input.png",])


#------------------------------------------------
# Pillow example:
# ------------------------------------------------
# Fully fledged test over rotations ran at 5 degree steps.

output_dir = "rotation_test/"
os.makedirs(output_dir, exist_ok=True)

angle_step = 5
angles = np.arange(0, 360, angle_step)

results_file = os.path.join(output_dir, "attitude_results.csv")

# Generate CSV labels
with open(results_file, "w") as f:
    f.write(f"Calculated Roll (deg), Expected Roll (deg)\n")


database_path = "roll-database.dat"

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


# Run through all rotations
for angle in angles:
    rotated_path = os.path.join(output_dir, f"rotated_{angle:03d}.png")
    annotated_path = os.path.join(output_dir, f"annotated_{angle:03d}.png")
    attitude_out = os.path.join(output_dir, f"attitude_{angle:03d}.txt")

    # Rotate image by N degrees
    img = Image.open("raw-input.png").rotate(angle)
    img.save(rotated_path)

    # Run through LOST
    result = subprocess.run([
        "./lost", "pipeline",
        "--png", rotated_path,
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
        "--print-attitude", attitude_out,
        "--plot-output", annotated_path
    ], capture_output=True, text=True)

    roll = None

    try:
        with open(attitude_out, "r") as f:
            for line in f:
                if line.startswith("attitude_roll"):
                    roll = float(line.split()[1])
                    break
    except Exception as e:
        print(f"Error reading attitude output for angle {angle}: {e}")
        roll = "NaN"


    
    with open(results_file, "a") as f:
        f.write(f"{angle},{str(roll)}\n")


df = pd.read_csv(results_file)


df.columns = [col.strip() for col in df.columns]


calculated = df["Calculated Roll (deg)"]
expected = df["Expected Roll (deg)"]


plt.figure(figsize=(8, 6))
plt.plot(expected, -calculated, "--", label="Recovered Roll", linewidth=1.8) # they have a negative correlation but thats arbitrary so we take the additive inverse of calculated
plt.plot(expected, expected, "--", color="gray", label="Expected Roll (y = x)", linewidth=1.2)


plt.xlabel("Expected Roll (deg)")
plt.ylabel("Calculated Roll (deg)")
plt.title("LOST Roll Recovery Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()


plt.savefig("rotation_test/roll_comparison.png")
