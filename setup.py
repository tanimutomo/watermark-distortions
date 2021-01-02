import setuptools import setup, find_packages

version = "0.0.1"
package_name = "distortion"

requirements = [
    "kornia>=0.4.0",
    "torch>=1.6.0",
]

if __name__ == "__main__":
    setup(
        name=package_name,
        version=version,
        author="Tomoki Tanimura",
        author_email="tanimutomo@gmail.com",
        url="https://github.com/tanimutomo/watermark-distortions",
        description="Watermark Distortions",
        license="Apache License 2.0",
        python_requires=">=3.6",

        # Test
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],

        # Package info
        packages=find_packages(exclude=('images', 'src/distortion_test.py')),

        install_requires=requirements,
    )