import setuptools

setuptools.setup(
    name="live_mocap",
    version="0.0.1",
    author="EasternJournalist@github.com",
    author_email="wangrc2081cs@mail.ustc.edu.cn",
    description="Cheap and easy live motion capture",
    long_description="Cheap and easy live motion capture",
    long_description_content_type="text/markdown",
    url="https://github.com/EasternJournalist/live_mocap",
    packages=['live_mocap'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        'opencv-python',
        'mediapipe',
        'scipy'
    ]
)