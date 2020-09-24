import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name='des-rl',
        version='0.1',
        scripts=['des-rl'],
        author="Federico Taschin",
        author_email="fedetask@gmail.com",
        description="De-Structured Reinforcement Learning algorithms",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/fedetask/des-rl",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
