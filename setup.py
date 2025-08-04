import os
from setuptools import setup, find_packages

readme_path = os.path.join(os.path.dirname(__file__), "README.md")

setup(
    name="gpt2-fine-tuned",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.44.0",
        "datasets>=2.0.0",
        "accelerate>=0.34.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.6.0",
        "flask>=2.3.0",
        "tqdm>=4.64.0",
        "requests>=2.28.0"
    ],
    entry_points={
        "console_scripts": [
            "generate-text=src.send_prompt:get_prompt",
        ],
    },
    author="Caleb Laurent",
    author_email="laurentcaleb99@gmail.com",
    description="A GPT-2 fine-tuned model for generating intelligent and humorous responses to jokes and riddles",
    long_description=open(readme_path, encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CalebLaurent509/gpt2_fine-tuned",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="gpt2, nlp, machine-learning, text-generation, humor, jokes, fine-tuning, transformers",
    python_requires='>=3.8',
    project_urls={
        "Bug Reports": "https://github.com/CalebLaurent509/gpt2_fine-tuned/issues",
        "Source": "https://github.com/CalebLaurent509/gpt2_fine-tuned",
        "Documentation": "https://github.com/CalebLaurent509/gpt2_fine-tuned/blob/main/README.md",
    },
)
