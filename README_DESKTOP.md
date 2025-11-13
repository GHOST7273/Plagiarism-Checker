# GHOST LLP - Text Intelligence Suite (Desktop Version)

A modern, aesthetic desktop application for paraphrasing text and detecting plagiarism using AI models.

## Features

- **✨ Modern Dark Theme UI**: Beautiful, interactive interface with smooth animations
- **Paraphrasing Tool**: Rewrite text while maintaining original meaning
- **Plagiarism Checker**: Check text against reference documents with detailed results
- **AI Detection**: Detect AI-generated content using perplexity analysis
- **Real-time Status Updates**: Visual indicators for model loading and processing status

## Installation

1. Make sure you have Python 3.8+ installed
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Desktop App

### Quick Start
**Windows:** Double-click `run_desktop.bat`

**Mac/Linux:** 
```bash
python desktop_app.py
```

Or use the launcher:
```bash
python run_desktop.py
```

### Creating a Desktop Shortcut (Windows)
```bash
pip install pywin32
python create_shortcut.py
```

This will create a shortcut on your desktop that you can double-click to launch the app.

### Building an Executable
To create a standalone .exe file (Windows):
```bash
pip install pyinstaller
python build_exe.py
```

The executable will be in the `dist` folder.

## First Run

On first run, the application will download the required AI models:
- T5 Paraphrase Model (~500MB)
- GPT-2 Model (~500MB)

This may take several minutes depending on your internet connection. The models will be cached for future use.

## Usage

### Paraphrasing Tool
1. Go to the "Paraphrasing Tool" tab
2. Enter or paste your text
3. Click "Paraphrase"
4. Copy the result using the "Copy to Clipboard" button

### Plagiarism Checker
1. Go to the "Plagiarism Checker" tab
2. Enter or paste the text you want to check
3. Click "Add File" to add reference documents (TXT or PDF)
4. Adjust thresholds if needed:
   - **TF-IDF Threshold**: Similarity threshold for snippet matching (0-1)
   - **AI Perplexity Threshold**: Lower values flag more text as AI-generated
5. Click "Check for Plagiarism"
6. Review the results in the right panel

## System Requirements

- **OS**: Windows, macOS, or Linux
- **RAM**: At least 4GB (8GB recommended)
- **Storage**: ~2GB free space for models
- **Python**: 3.8 or higher

## Troubleshooting

### Models not loading
- Check your internet connection (required for first-time download)
- Ensure you have enough disk space
- Try running with administrator/sudo privileges

### Application is slow
- The first run downloads models (one-time)
- Processing large texts may take time
- Close other applications to free up memory

### GUI not appearing
- Make sure tkinter is installed (usually comes with Python)
- On Linux, you may need to install: `sudo apt-get install python3-tk`

## Notes

- The desktop app runs entirely locally - no internet required after initial model download
- All processing happens on your computer - your data stays private
- Models are cached in your user directory (typically `~/.cache/huggingface/`)

