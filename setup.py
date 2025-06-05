from setuptools import setup, find_packages

setup(
    name="dll",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pyyaml",
        "tqdm",
        "pillow",
        "numpy",
        "torchvision",
        "opencv-python",
        "torchaudio",
        "scipy"
    ]
)