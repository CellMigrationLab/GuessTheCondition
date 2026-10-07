from setuptools import setup, find_packages

setup(
    name="guessthecondition",
    version="0.1.0",
    description="Guess the Condition: can the experimental conditions be told apart by looking at the images?",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "tifffile",
    ],
)
