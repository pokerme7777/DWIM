from setuptools import setup, find_packages  # type: ignore

# setup(
#     name="study_on_agent",
#     version="0.1",
#     packages=["src", "neurips_prototyping"],
#     install_requires=[],
# )
setup(
    name="study_on_agent",
    version="0.1",
    packages=find_packages(include=["src*", "neurips_prototyping*"]),
    install_requires=[],
)