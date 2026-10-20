# Automated Answer Script Evaluation System

A Windows-local application that evaluates student answer scripts using OCR and a local language model.

## Workflow

1. Upload a question paper and a marking scheme/rubric (or let the system draft a rubric).
2. Upload scanned or photographed student answer sheets.
3. OCR extracts the answers.
4. The system matches answers to questions and evaluates them against rubric criteria.
5. It creates reports and supports teacher review.
6. An AI-vs-human agreement tab compares AI marks with faculty marks.

## Repository files

- `main.py` — Python application converted from the notebook's code cells, kept in their original execution order.
- `requirements.txt` — Python dependencies.
- `setup_windows.bat` — creates a virtual environment and installs dependencies.
- `.gitignore` — excludes local environments, model files, databases, uploads, and generated reports.
- `Answer_Script_Evaluator_Windows_Local.ipynb` — original notebook, included for reference.

## Requirements

- Windows 10/11 (the notebook is designed for Windows).
- Python 3.11 or 3.12 recommended.
- NVIDIA GPU and a compatible NVIDIA driver recommended. CPU fallback may be very slow.
- Around 15 GB or more free disk space for model downloads and runtime data.
- Internet access for initial dependency/model downloads. The application is designed to run locally after setup and model downloads.

## Setup

1. Install Python 3.11 or 3.12 and an up-to-date NVIDIA driver if using an NVIDIA GPU.
2. Double-click `setup_windows.bat` and wait for it to finish.
3. Open Command Prompt in this repository folder and activate the environment:

   ```bat
   .venv\Scripts\activate
   ```

4. Run the app:

   ```bat
   python main.py
   ```

The first run downloads large model files and may take a while. The application is configured to use `BASE_DIR` (`D:\AnswerEvalAI` if the D: drive exists, otherwise `C:\AnswerEvalAI`) for model files, data, and reports. Change `BASE_DIR` near the top of `main.py` if required.

The Gradio interface is configured to bind to `127.0.0.1` and does not enable public sharing.

## Important notes

- This is a local research/academic project, not a substitute for a teacher's final judgment. Review AI-generated marks and feedback before using them.
- Keep student answer sheets and personally identifying information out of a public GitHub repository.
- Do not upload model weights, `.venv`, local databases, student uploads, generated reports, or API tokens to GitHub.
- `llama-cpp-python` installation on Windows can require a compatible prebuilt wheel or additional build configuration. The application also attempts to use a downloaded `llama-server.exe` backend.
- The generated `main.py` preserves the notebook code-cell order. It has been syntax-checked, but the full application has not been run end-to-end on your computer; hardware, driver, package-version, and model-download differences may require adjustments.

## Project description

This project is a locally deployable AI-based answer-script evaluation system. It combines document/image preprocessing, vision-language OCR, rubric-based question-wise grading, feedback generation, report export, teacher review, and agreement metrics between AI and human marks.
