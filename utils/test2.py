def check_package(package_name, display_name=None):
    try:
        __import__(package_name)
        return f"{display_name or package_name:<15}: ✅ Installed"
    except ImportError:
        return f"{display_name or package_name:<15}: ❌ Not Installed"

def main():
    packages = {
        "numpy": "NumPy",
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        "scipy": "SciPy",
        "sklearn": "scikit-learn",
        "cv2": "OpenCV",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "jax": "JAX",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost"
    }

    print("Checking installed packages...\n")
    for pkg, name in packages.items():
        print(check_package(pkg, name))

if __name__ == "__main__":
    main()
