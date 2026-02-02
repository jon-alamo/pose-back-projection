import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bppose",
    version="0.1.0",
    author="Joni",
    description="A 3D pose back-projection library for Padel players using VideoPose3D",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "pandas",
        "torch",
    ],
    package_data={
        "bppose": ["checkpoints/*.bin"],
    },
    include_package_data=True,
)
