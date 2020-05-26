import setuptools

setuptools.setup(
    name="dl-selection",
    version="0.0.3",
    author="Ian Covert",
    author_email="icovert@cs.washington.edu",
    description="Feature selection for deep learning models.",
    long_description="""
        The **dl-selection** package contains tools for performing feature 
        selection with deep learning models. It supports several input layers 
        for selecting features, each of which relies on a stochastic relaxation 
        of the feature selection problem. See the 
        [GitHub page](https://github.com/icc2115/dl-selection/) for more 
        details.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/icc2115/dl-selection/",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'torch'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
