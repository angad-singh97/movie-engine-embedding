from setuptools import setup, find_packages

setup(
    name="movie_query_engine",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas", "numpy", "scikit-learn", "sentence-transformers", "tqdm", "matplotlib", "seaborn"
    ],
    entry_points={
        'console_scripts': [
            'movie-query-cli=movie_query_engine.engine:interactive_cli_tfidf_full'
        ]
    },
    author="Angad S",
    description="Search engine over movies using TF-IDF and SBERT",
    python_requires=">=3.7",
)
