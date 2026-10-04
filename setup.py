from setuptools import setup, find_packages

setup(
    name="kolibriforge",
    version="1.0.0",
    description="Sovereign MoE reasoning LLM with honest abstention (Kolibri-1 direction)",
    packages=find_packages(include=["kolibriforge*"]),
    python_requires=">=3.9",
    entry_points={"console_scripts": ["kolibriforge=kolibriforge.cli.main:main"]},
)
