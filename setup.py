from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="movie_query_engine",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=2.2.2",
        "numpy>=1.24.4",
        "scikit-learn>=1.3.2",
        "torch>=1.13,<2.1",
        "sentence-transformers>=2.2.2",
        "huggingface-hub>=0.27.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.2"
    ],
    entry_points={
        'console_scripts': [
            'movie-query-cli=movie_query_engine.engine:interactive_cli_tfidf_full'
        ]
    },
    author="Angad S",
    description="Search engine over movies using TF-IDF and SBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
