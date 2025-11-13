# GHOST LLP - Text Intelligence Suite (Desktop Version)

A modern, aesthetic desktop application for paraphrasing text and detecting plagiarism using AI models.

## Features

- **🏠 Start Menu**: Beautiful home screen to choose your tool
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

## First Run

On first run, the application will:
1. Show a Start Menu with two options:
   - Paraphrasing Tool
   - Plagiarism Checker
2. Download the required AI models:
   - T5 Paraphrase Model (~500MB)
   - GPT-2 Model (~500MB)

This may take several minutes depending on your internet connection. The models will be cached for future use.

## Usage

### Start Menu
When you launch the app, you'll see a Start Menu with two large buttons:
- Click "Paraphrasing Tool" to rewrite text
- Click "Plagiarism Checker" to check for plagiarism
- Use "← Back to Start" button to return to the main menu

### Paraphrasing Tool
1. Click "Paraphrasing Tool" from the Start Menu
2. Enter or paste your text
3. Click "Paraphrase"
4. Copy the result using the "Copy to Clipboard" button

### Plagiarism Checker
1. Click "Plagiarism Checker" from the Start Menu
2. Enter or paste the text you want to check
3. Click "Add File" to add reference documents (TXT or PDF)
4. Adjust thresholds if needed:
   - **TF-IDF Threshold**: Similarity threshold for snippet matching (0-1)
   - **AI Perplexity Threshold**: Lower values flag more text as AI-generated
5. Click "Check for Plagiarism"
6. Review the results in the right panel
7. Results will appear automatically after processing

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

### Paraphrasing not working
- Make sure models are fully loaded (check status indicator)
- Wait for the "Models loaded! Ready to use." message
- Try entering shorter text first

## Notes

- The desktop app runs entirely locally - no internet required after initial model download
- All processing happens on your computer - your data stays private
- Models are cached in your user directory (typically `~/.cache/huggingface/`)
- You can navigate between tools using the "Back to Start" button

