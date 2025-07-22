import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "numpy", "scipy", "pandas", "matplotlib", "seaborn",
    "scikit-learn", "tensorflow", "torch", "torchvision", "torchaudio",
    "opencv-python", "xgboost", "lightgbm", "catboost",
    "jax", "jaxlib", "transformers", "datasets",
    "nltk", "spacy", "statsmodels", "sympy", "plotly",
    "keras", "tqdm"
]

print("Installing packages...")
for pkg in packages:
    try:
        install(pkg)
        print(f"✅ {pkg} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {pkg}")

print("Installation complete.")
