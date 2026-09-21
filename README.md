# Handwritten Marks Extraction

A Python/OpenCV-based project for extracting handwritten student information and marks from standardized examination answer sheets.

---

# Setup Guide

Follow the steps below to set up the project on a new computer.

## 1. Check Whether Python Is Installed

Before starting, check whether Python is already installed.

Open **Command Prompt** or the **VS Code Terminal** and run:

```bash
python --version
```

=============================================================
========================MAIN WORK STARTS HERE================

# Handwritten Marks Extraction

A Python/OpenCV-based project for extracting handwritten student information and marks from examination answer sheets.

## Python Environment Setup

The virtual environment (`.venv`) is **not included in this GitHub repository**.

Each developer should create their own virtual environment after cloning the repository.

### 1. Clone the repository

```bash
git clone https://github.com/Debastab0-2107/Handwritten-marks-extraction.git
```

Go into the project folder:

```bash
cd Handwritten-marks-extraction
```

### 2. Create a virtual environment

Make sure Python is installed.

Run:

```bash
python -m venv .venv
```

This creates a local virtual environment named `.venv`.

### 3. Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

After activation, the terminal should show something similar to:

```text
(.venv) C:\...\Handwritten-marks-extraction>
```

### 4. Install the project dependencies

The required Python packages are listed in `requirements.txt`.

Run:

```bash
pip install -r requirements.txt
```

### 5. Run the project

After activating the virtual environment and installing the dependencies:

```bash
python main.py
```

## Current Supported Image Formats

The current image-processing pipeline supports:

* `.jpg`
* `.jpeg`
* `.png`

## PDF Support

PDF processing is planned for a later stage of development.

PDF-related dependencies such as `pdf2image` are **not required at the current stage**.

## Important

Do **not** commit the `.venv` folder to GitHub.

The `.venv` folder is specific to each developer's computer.

Each developer should create their own environment using:

```bash
python -m venv .venv
```

and then install the dependencies using:

```bash
pip install -r requirements.txt
```
