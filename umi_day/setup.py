from setuptools import setup, find_packages

INSTALL_REQUIRES = []

setup(
    name="umi_day",
    author="Austin Patel",
    version="1.0.0",
    description="",
    keywords=[],
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=[],
    zip_safe=False
)
