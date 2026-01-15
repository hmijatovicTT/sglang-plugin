"""
SGLang TT-Metal Plugin Setup

IMPORTANT INSTALLATION ORDER:
1. Install CPU PyTorch first: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
2. Install sglang: pip install sglang
3. Install this plugin: pip install -e .

This ensures sgl_kernel and common_ops are built for CPU, not GPU.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sglang-tt-plugin",
    version="0.1.0",
    author="TT-Metal Integration Team",
    author_email="your-email@example.com",
    description="Tenstorrent TT plugin for SGLang",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/sglang-tt-plugin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        # NOTE: Do NOT include torch here - user must install CPU PyTorch FIRST
        # to ensure sgl_kernel builds for CPU. See installation instructions.
        # "sglang",  # Using source code version instead of pip
        "transformers",
        # ttnn should be available from tt-metal environment
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-asyncio",
        ],
    },
    entry_points={
        "console_scripts": [
            "sglang-tt-server = sglang_tt_plugin.scripts.launch_tt_server:main",
        ],
        "sglang.models": [
            "tt_llama = sglang_tt_plugin.models.tt_llama:TTLlamaForCausalLM",
        ],
    },
    package_data={
        "sglang_tt_plugin": ["*.py"],
    },
)