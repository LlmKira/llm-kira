import setuptools

with open("README.md", "r") as fh:
    _LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="llm-kira",
    license='LGPL-2.1-or-later',
    author="sudoskys",
    version="0.0.1",
    author_email="me@dianas.cyou",
    description="LLM client",
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/sudoskys/llm-kira",
    packages=setuptools.find_namespace_packages(),
    install_requires=[
        "httpx",
        "redis",
        "nltk",
        "Pillow",
        "numpy",
        "jieba",
        "transformers",
        "beautifulsoup4",
        "pydantic",
        "loguru",
        "elara"
    ],
)
