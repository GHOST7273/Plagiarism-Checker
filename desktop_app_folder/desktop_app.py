"""
GHOST LLP - Text Intelligence Suite Desktop Application
Modern desktop app with Start menu
"""

import os
import io
import math
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from datetime import datetime
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
import difflib
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

# Ensure NLTK tokenizer resources are available
for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        if _resource == "punkt":
            raise

# ---------------------------
# Global Variables
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paraphrase_tokenizer = None
paraphrase_model = None
tokenizer_gpt2 = None
model_gpt2 = None
models_loaded = False

# Color scheme - Modern and aesthetic
COLORS = {
    'bg_primary': '#1a1a2e',
    'bg_secondary': '#16213e',
    'bg_tertiary': '#0f3460',
    'accent_blue': '#2196F3',
    'accent_green': '#4CAF50',
    'accent_red': '#f44336',
    'accent_orange': '#FF9800',
    'text_primary': '#ffffff',
    'text_secondary': '#b0b0b0',
    'card_bg': '#1e2749',
    'border': '#2a3d66'
}

# ---------------------------
# Core Functions
# ---------------------------
def paraphrase_text(sentence):
    """Paraphrase a given sentence using the T5 model."""
    if not models_loaded or paraphrase_tokenizer is None or paraphrase_model is None:
        raise Exception("Models not loaded yet. Please wait...")
    
    text = "paraphrase: " + sentence + " </s>"
    encoding = paraphrase_tokenizer.encode_plus(text, padding='longest', return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    outputs = paraphrase_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
        temperature=1.0,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    paraphrased = paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

def load_text_from_file(filepath):
    """Load text from a file (supports PDF and text files)."""
    if filepath.lower().endswith('.pdf'):
        return extract_text(filepath)
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def compute_tfidf_similarities(query_text, reference_texts):
    """Compute TF-IDF similarities between query and reference texts."""
    corpus = [query_text] + reference_texts
    vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
    tfidf = vectorizer.fit_transform(corpus)
    query_vec = tfidf[0]
    refs = tfidf[1:]
    sims = cosine_similarity(query_vec, refs)[0]
    return sims

def get_matching_snippets(query, reference, cutoff=0.6, max_snippets=5):
    """Find matching snippets between query and reference."""
    matcher = difflib.SequenceMatcher(None, query, reference)
    blocks = matcher.get_matching_blocks()
    snippets = []
    
    sorted_blocks = sorted(blocks, key=lambda b: b[2], reverse=True)
    
    for b in sorted_blocks:
        i, j, size = b
        if size <= 20:
            continue
        
        snippet_text = query[i:i+size].strip()
        if not snippet_text:
            continue
            
        snippet_matcher = difflib.SequenceMatcher(None, snippet_text, reference)
        ratio = snippet_matcher.ratio()
        
        if ratio < cutoff:
            continue
        
        start = max(0, i-40)
        end = min(len(query), i+size+40)
        context = query[start:end].replace('\n', ' ')
        snippets.append({
            'match': snippet_text,
            'context': context,
            'match_length': size,
            'similarity': ratio
        })
        if len(snippets) >= max_snippets:
            break
    return snippets

def compute_exact_coverage(query, reference):
    """Compute exact character coverage between query and reference."""
    if not query or not reference:
        return 0.0
    
    q = query
    ref = reference
    total_chars = len(q)
    
    window = 20
    step = 5
    i = 0
    matched_positions = set()
    
    while i < len(q):
        for win_size in [window, window + 10, window + 20]:
            if i + win_size > len(q):
                continue
            segment = q[i:i+win_size].strip()
            if segment and len(segment) >= 10:
                if segment in ref:
                    for pos in range(i, min(i + win_size, len(q))):
                        matched_positions.add(pos)
                    i += step
                    break
        else:
            i += step
    
    matched_chars = len(matched_positions)
    if total_chars == 0:
        return 0.0
    
    return matched_chars / total_chars

def perplexity_of_text(text):
    """Compute perplexity of text using GPT-2."""
    if not models_loaded or tokenizer_gpt2 is None or model_gpt2 is None:
        return 100.0
    
    enc = tokenizer_gpt2(text, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(device)
    with torch.no_grad():
        outputs = model_gpt2(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    ppl = math.exp(loss) if loss < 50 else float('inf')
    return ppl

def ai_sentence_flags(text, threshold=40.0):
    """Flag sentences as AI-generated based on perplexity."""
    sents = sent_tokenize(text)
    results = []
    for s in sents:
        p = perplexity_of_text(s)
        flagged = p < threshold
        results.append({'sentence': s, 'perplexity': p, 'flagged': flagged})
    return results

# ---------------------------
# Model Loading
# ---------------------------
def load_models():
    """Load ML models in a separate thread."""
    global paraphrase_tokenizer, paraphrase_model, tokenizer_gpt2, model_gpt2, models_loaded
    
    try:
        print("Loading paraphrasing model...")
        paraphrase_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", use_fast=False)
        paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        
        print("Loading GPT-2 model for AI detection...")
        gpt2_model_name = "gpt2"
        tokenizer_gpt2 = AutoTokenizer.from_pretrained(gpt2_model_name)
        model_gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(device)
        model_gpt2.eval()
        
        models_loaded = True
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# ---------------------------
# Modern Styled Widgets
# ---------------------------
class ModernButton(tk.Button):
    """Modern styled button with hover effects."""
    def __init__(self, parent, text, command, bg_color=COLORS['accent_blue'], **kwargs):
        self.bg_color = bg_color
        self.hover_color = self._lighten_color(bg_color)
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            activebackground=self.hover_color,
            activeforeground=COLORS['text_primary'],
            **kwargs
        )
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
    
    def _on_enter(self, e):
        self.config(bg=self.hover_color)
    
    def _on_leave(self, e):
        self.config(bg=self.bg_color)
    
    def _lighten_color(self, color):
        """Lighten a hex color."""
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        rgb = tuple(min(255, int(c * 1.2)) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

class ModernFrame(tk.Frame):
    """Modern styled frame."""
    def __init__(self, parent, **kwargs):
        bg = kwargs.pop('bg', COLORS['card_bg'])
        super().__init__(parent, bg=bg, **kwargs)

# ---------------------------
# Desktop Application GUI
# ---------------------------
class TextIntelligenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GHOST LLP - Text Intelligence Suite")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLORS['bg_primary'])
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=COLORS['bg_primary'], borderwidth=0)
        style.configure('TNotebook.Tab', background=COLORS['bg_secondary'], foreground=COLORS['text_primary'],
                        padding=[20, 10], font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', COLORS['accent_blue'])])
        
        # Create main container
        self.main_frame = tk.Frame(root, bg=COLORS['bg_primary'])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Current view tracker
        self.current_view = "start"
        
        # Create Start Menu
        self.create_start_menu()
        
        # Load models in background
        self.load_models_thread()
    
    def load_models_thread(self):
        """Load models in a separate thread to avoid freezing the GUI."""
        def load():
            success = load_models()
            if success:
                self.root.after(0, lambda: self.update_start_status("Models loaded! Ready to use.", 'green'))
            else:
                self.root.after(0, lambda: self.update_start_status("Error loading models. Some features may not work.", 'red'))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def create_start_menu(self):
        """Create the start/home menu screen."""
        # Clear existing widgets
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        self.current_view = "start"
        
        # Header
        header_frame = tk.Frame(self.main_frame, bg=COLORS['bg_secondary'], height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_content = tk.Frame(header_frame, bg=COLORS['bg_secondary'])
        header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        title_label = tk.Label(header_content, text="GHOST LLP", 
                              font=('Segoe UI', 32, 'bold'), 
                              bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        title_label.pack()
        
        subtitle_label = tk.Label(header_content, text="Text Intelligence Suite", 
                                   font=('Segoe UI', 14), 
                                   bg=COLORS['bg_secondary'], fg=COLORS['text_secondary'])
        subtitle_label.pack(pady=(5, 0))
        
        # Status indicator
        self.status_indicator = tk.Label(header_content, text="●", 
                                         font=('Segoe UI', 16),
                                         bg=COLORS['bg_secondary'], fg=COLORS['accent_orange'])
        self.status_indicator.pack(pady=(10, 0))
        
        self.status_label = tk.Label(header_content, text="Loading models...", 
                                     bg=COLORS['bg_secondary'], fg=COLORS['text_secondary'],
                                     font=('Segoe UI', 10))
        self.status_label.pack()
        
        # Main content area
        content_frame = tk.Frame(self.main_frame, bg=COLORS['bg_primary'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=50)
        
        # Welcome message
        welcome_card = ModernFrame(content_frame, bg=COLORS['card_bg'], relief=tk.FLAT, bd=2)
        welcome_card.pack(fill=tk.BOTH, expand=True, pady=(0, 30))
        
        welcome_text = tk.Label(welcome_card, 
                               text="Welcome to GHOST LLP Text Intelligence Suite",
                               font=('Segoe UI', 18, 'bold'),
                               bg=COLORS['card_bg'], fg=COLORS['text_primary'])
        welcome_text.pack(pady=30)
        
        desc_text = tk.Label(welcome_card,
                            text="Choose a tool to get started:",
                            font=('Segoe UI', 12),
                            bg=COLORS['card_bg'], fg=COLORS['text_secondary'])
        desc_text.pack(pady=(0, 40))
        
        # Tool selection buttons
        button_container = tk.Frame(welcome_card, bg=COLORS['card_bg'])
        button_container.pack(pady=(0, 40))
        
        # Paraphrasing Tool Button
        paraphrase_btn = ModernButton(button_container, 
                                      "✨ Paraphrasing Tool\n\nRewrite text while maintaining meaning",
                                      lambda: self.show_paraphrase_tool(),
                                      bg_color=COLORS['accent_blue'])
        paraphrase_btn.config(font=('Segoe UI', 12, 'bold'), width=25, height=3)
        paraphrase_btn.pack(side=tk.LEFT, padx=20)
        
        # Plagiarism Checker Button
        plagiarism_btn = ModernButton(button_container,
                                      "🔍 Plagiarism Checker\n\nCheck text against references",
                                      lambda: self.show_plagiarism_tool(),
                                      bg_color=COLORS['accent_green'])
        plagiarism_btn.config(font=('Segoe UI', 12, 'bold'), width=25, height=3)
        plagiarism_btn.pack(side=tk.LEFT, padx=20)
    
    def update_start_status(self, message, color):
        """Update status label and indicator."""
        if hasattr(self, 'status_label') and self.current_view == "start":
            self.status_label.config(text=message, fg=color)
            self.status_indicator.config(fg=color)
    
    def show_paraphrase_tool(self):
        """Show the paraphrasing tool interface."""
        # Clear existing widgets
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        self.current_view = "paraphrase"
        
        # Header with back button
        header_frame = tk.Frame(self.main_frame, bg=COLORS['bg_secondary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_content = tk.Frame(header_frame, bg=COLORS['bg_secondary'])
        header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)
        
        back_btn = ModernButton(header_content, "← Back to Start", 
                               self.create_start_menu,
                               bg_color=COLORS['bg_tertiary'])
        back_btn.config(font=('Segoe UI', 9))
        back_btn.pack(side=tk.LEFT)
        
        title_label = tk.Label(header_content, text="Paraphrasing Tool", 
                               font=('Segoe UI', 20, 'bold'), 
                               bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Container with padding
        container = ModernFrame(self.main_frame, bg=COLORS['bg_primary'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Input section
        input_card = ModernFrame(container, bg=COLORS['card_bg'], relief=tk.FLAT, bd=2)
        input_card.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        input_header = tk.Frame(input_card, bg=COLORS['card_bg'], height=50)
        input_header.pack(fill=tk.X)
        input_header.pack_propagate(False)
        
        tk.Label(input_header, text="Enter Text to Paraphrase", 
                font=('Segoe UI', 14, 'bold'),
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(side=tk.LEFT, padx=20, pady=15)
        
        input_body = tk.Frame(input_card, bg=COLORS['card_bg'])
        input_body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.paraphrase_input = scrolledtext.ScrolledText(
            input_body, 
            height=12, 
            font=('Segoe UI', 11),
            wrap=tk.WORD,
            bg='#2a3d66',
            fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            selectbackground=COLORS['accent_blue'],
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=15
        )
        self.paraphrase_input.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(input_card, bg=COLORS['card_bg'])
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        paraphrase_btn = ModernButton(button_frame, "Paraphrase", 
                                     self.handle_paraphrase,
                                     bg_color=COLORS['accent_blue'])
        paraphrase_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ModernButton(button_frame, "Clear", 
                               self.clear_paraphrase,
                               bg_color=COLORS['bg_tertiary'])
        clear_btn.pack(side=tk.LEFT)
        
        # Output section
        output_card = ModernFrame(container, bg=COLORS['card_bg'], relief=tk.FLAT, bd=2)
        output_card.pack(fill=tk.BOTH, expand=True)
        
        output_header = tk.Frame(output_card, bg=COLORS['card_bg'], height=50)
        output_header.pack(fill=tk.X)
        output_header.pack_propagate(False)
        
        tk.Label(output_header, text="Paraphrased Text", 
                font=('Segoe UI', 14, 'bold'),
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(side=tk.LEFT, padx=20, pady=15)
        
        output_body = tk.Frame(output_card, bg=COLORS['card_bg'])
        output_body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.paraphrase_output = scrolledtext.ScrolledText(
            output_body, 
            height=12, 
            font=('Segoe UI', 11),
            wrap=tk.WORD,
            bg='#2a3d66',
            fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            selectbackground=COLORS['accent_blue'],
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=15,
            state=tk.DISABLED
        )
        self.paraphrase_output.pack(fill=tk.BOTH, expand=True)
        
        # Copy button
        copy_btn = ModernButton(output_body, "📋 Copy to Clipboard", 
                               self.copy_paraphrase,
                               bg_color=COLORS['accent_green'])
        copy_btn.pack(pady=10)
    
    def show_plagiarism_tool(self):
        """Show the plagiarism checker interface."""
        # Clear existing widgets
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        self.current_view = "plagiarism"
        
        # Header with back button
        header_frame = tk.Frame(self.main_frame, bg=COLORS['bg_secondary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_content = tk.Frame(header_frame, bg=COLORS['bg_secondary'])
        header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)
        
        back_btn = ModernButton(header_content, "← Back to Start", 
                               self.create_start_menu,
                               bg_color=COLORS['bg_tertiary'])
        back_btn.config(font=('Segoe UI', 9))
        back_btn.pack(side=tk.LEFT)
        
        title_label = tk.Label(header_content, text="Plagiarism Checker", 
                              font=('Segoe UI', 20, 'bold'), 
                              bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Main container
        main_container = tk.Frame(self.main_frame, bg=COLORS['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg=COLORS['bg_primary'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Query input card
        query_card = ModernFrame(left_panel, bg=COLORS['card_bg'], relief=tk.FLAT, bd=2)
        query_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        tk.Label(query_card, text="Text to Check", 
                font=('Segoe UI', 12, 'bold'),
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(anchor=tk.W, padx=20, pady=(15, 10))
        
        self.plagiarism_input = scrolledtext.ScrolledText(
            query_card, 
            height=15, 
            font=('Segoe UI', 10),
            wrap=tk.WORD,
            bg='#2a3d66',
            fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            selectbackground=COLORS['accent_blue'],
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=15
        )
        self.plagiarism_input.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Reference files card
        ref_card = ModernFrame(left_panel, bg=COLORS['card_bg'], relief=tk.FLAT, bd=2)
        ref_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(ref_card, text="Reference Files", 
                font=('Segoe UI', 12, 'bold'),
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(anchor=tk.W, padx=20, pady=(15, 10))
        
        # Listbox with scrollbar
        listbox_frame = tk.Frame(ref_card, bg=COLORS['card_bg'])
        listbox_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.ref_listbox = tk.Listbox(
            listbox_frame, 
            height=4, 
            font=('Segoe UI', 9),
            bg='#2a3d66',
            fg=COLORS['text_primary'],
            selectbackground=COLORS['accent_blue'],
            relief=tk.FLAT,
            bd=0,
            yscrollcommand=scrollbar.set
        )
        self.ref_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.ref_listbox.yview)
        
        self.ref_files = []
        
        ref_btn_frame = tk.Frame(ref_card, bg=COLORS['card_bg'])
        ref_btn_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        add_ref_btn = ModernButton(ref_btn_frame, "➕ Add File", 
                                   self.add_reference_file,
                                   bg_color=COLORS['accent_blue'])
        add_ref_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        remove_ref_btn = ModernButton(ref_btn_frame, "➖ Remove", 
                                      self.remove_reference_file,
                                      bg_color=COLORS['accent_red'])
        remove_ref_btn.pack(side=tk.LEFT)
        
        # Settings card
        settings_card = ModernFrame(left_panel, bg=COLORS['card_bg'], relief=tk.FLAT, bd=2)
        settings_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(settings_card, text="Settings", 
                font=('Segoe UI', 12, 'bold'),
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(anchor=tk.W, padx=20, pady=(15, 10))
        
        settings_body = tk.Frame(settings_card, bg=COLORS['card_bg'])
        settings_body.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(settings_body, text="TF-IDF Threshold (0-1):", 
                bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(0, 5))
        self.tfidf_thresh = tk.Entry(settings_body, font=('Segoe UI', 10),
                                     bg='#2a3d66', fg=COLORS['text_primary'],
                                     insertbackground=COLORS['text_primary'],
                                     relief=tk.FLAT, bd=0)
        self.tfidf_thresh.insert(0, "0.4")
        self.tfidf_thresh.pack(fill=tk.X, pady=(0, 15), ipady=8)
        
        tk.Label(settings_body, text="AI Perplexity Threshold:", 
                bg=COLORS['card_bg'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(0, 5))
        self.ppl_thresh = tk.Entry(settings_body, font=('Segoe UI', 10),
                                   bg='#2a3d66', fg=COLORS['text_primary'],
                                   insertbackground=COLORS['text_primary'],
                                   relief=tk.FLAT, bd=0)
        self.ppl_thresh.insert(0, "40.0")
        self.ppl_thresh.pack(fill=tk.X, ipady=8)
        
        # Check button
        check_btn = ModernButton(left_panel, "🔍 Check for Plagiarism", 
                               self.handle_plagiarism_check,
                               bg_color=COLORS['accent_green'])
        check_btn.pack(fill=tk.X, pady=10)
        
        # Right panel - Results
        right_panel = tk.Frame(main_container, bg=COLORS['bg_primary'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        results_card = ModernFrame(right_panel, bg=COLORS['card_bg'], relief=tk.FLAT, bd=2)
        results_card.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(results_card, text="Results", 
                font=('Segoe UI', 12, 'bold'),
                bg=COLORS['card_bg'], fg=COLORS['text_primary']).pack(anchor=tk.W, padx=20, pady=(15, 10))
        
        self.plagiarism_results = scrolledtext.ScrolledText(
            results_card, 
            font=('Segoe UI', 9),
            wrap=tk.WORD,
            bg='#2a3d66',
            fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            selectbackground=COLORS['accent_blue'],
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=15,
            state=tk.DISABLED
        )
        self.plagiarism_results.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
    
    def handle_paraphrase(self):
        """Handle paraphrasing request."""
        text = self.paraphrase_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to paraphrase.")
            return
        
        if not models_loaded:
            messagebox.showwarning("Warning", "Models are still loading. Please wait...")
            return
        
        # Update UI to show processing
        self.paraphrase_output.config(state=tk.NORMAL)
        self.paraphrase_output.delete("1.0", tk.END)
        self.paraphrase_output.insert("1.0", "Processing... Please wait.")
        self.paraphrase_output.config(state=tk.DISABLED)
        self.root.update()
        
        def paraphrase():
            try:
                result = paraphrase_text(text)
                self.root.after(0, lambda r=result: self.display_paraphrase_result(r))
            except Exception as e:
                error_str = str(e)
                self.root.after(0, lambda err=error_str: self.display_error(err))
        
        thread = threading.Thread(target=paraphrase, daemon=True)
        thread.start()
    
    def display_paraphrase_result(self, result):
        """Display paraphrasing result."""
        self.paraphrase_output.config(state=tk.NORMAL)
        self.paraphrase_output.delete("1.0", tk.END)
        self.paraphrase_output.insert("1.0", result)
        self.paraphrase_output.config(state=tk.DISABLED)
    
    def display_error(self, error_msg):
        """Display error message."""
        self.paraphrase_output.config(state=tk.NORMAL)
        self.paraphrase_output.delete("1.0", tk.END)
        self.paraphrase_output.insert("1.0", f"Error: {error_msg}")
        self.paraphrase_output.config(state=tk.DISABLED)
        messagebox.showerror("Error", error_msg)
    
    def clear_paraphrase(self):
        """Clear paraphrasing input and output."""
        self.paraphrase_input.delete("1.0", tk.END)
        self.paraphrase_output.config(state=tk.NORMAL)
        self.paraphrase_output.delete("1.0", tk.END)
        self.paraphrase_output.config(state=tk.DISABLED)
    
    def copy_paraphrase(self):
        """Copy paraphrased text to clipboard."""
        text = self.paraphrase_output.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Success", "Text copied to clipboard!")
    
    def add_reference_file(self):
        """Add a reference file for plagiarism checking."""
        filepath = filedialog.askopenfilename(
            title="Select Reference File",
            filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filepath:
            self.ref_files.append(filepath)
            filename = os.path.basename(filepath)
            self.ref_listbox.insert(tk.END, filename)
    
    def remove_reference_file(self):
        """Remove selected reference file."""
        selection = self.ref_listbox.curselection()
        if selection:
            index = selection[0]
            self.ref_listbox.delete(index)
            self.ref_files.pop(index)
    
    def handle_plagiarism_check(self):
        """Handle plagiarism check request."""
        query = self.plagiarism_input.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter text to check.")
            return
        
        if not self.ref_files:
            messagebox.showwarning("Warning", "Please add at least one reference file.")
            return
        
        # Update UI
        self.plagiarism_results.config(state=tk.NORMAL)
        self.plagiarism_results.delete("1.0", tk.END)
        self.plagiarism_results.insert("1.0", "Processing... Please wait.\n\nThis may take a moment...")
        self.plagiarism_results.config(state=tk.DISABLED)
        self.root.update()
        
        def check():
            try:
                tfidf_thresh = float(self.tfidf_thresh.get())
                ppl_thresh = float(self.ppl_thresh.get())
                
                # Load reference texts
                reference_texts = []
                ref_names = []
                for filepath in self.ref_files:
                    try:
                        text = load_text_from_file(filepath)
                        reference_texts.append(text)
                        ref_names.append(os.path.basename(filepath))
                    except Exception as e:
                        error_msg = f"Error loading file {os.path.basename(filepath)}: {str(e)}"
                        self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
                        return
                
                # Compute similarities
                sims = compute_tfidf_similarities(query, reference_texts) if reference_texts else []
                max_tfidf = 0.0
                coverage_scores = []
                per_reference = []
                
                for idx, ref_name in enumerate(ref_names):
                    tfidf_score = float(sims[idx]) if len(sims) > idx else 0.0
                    if tfidf_score > max_tfidf:
                        max_tfidf = tfidf_score
                    
                    coverage = compute_exact_coverage(query, reference_texts[idx])
                    coverage_scores.append(coverage)
                    snippets = get_matching_snippets(query, reference_texts[idx], cutoff=tfidf_thresh)
                    
                    per_reference.append({
                        'filename': ref_name,
                        'tfidf': tfidf_score,
                        'coverage': coverage,
                        'snippets': snippets
                    })
                
                avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
                max_coverage = max(coverage_scores) if coverage_scores else 0.0
                
                estimated = 0.5 * max_tfidf + 0.3 * avg_coverage + 0.2 * max_coverage
                estimated = min(estimated, 1.0)
                
                # AI detection
                ai_flags = ai_sentence_flags(query, threshold=ppl_thresh)
                total_sentences = len(ai_flags)
                ai_flagged_count = sum(1 for item in ai_flags if item['flagged'])
                ai_flagged_percent = (ai_flagged_count / total_sentences) if total_sentences else 0.0
                
                # Format results
                result_text = self.format_plagiarism_results(
                    per_reference, max_tfidf, avg_coverage, max_coverage, 
                    estimated, ai_flags, ppl_thresh, ai_flagged_count, 
                    ai_flagged_percent, total_sentences
                )
                
                # Use lambda with default argument to avoid closure issues
                self.root.after(0, lambda rt=result_text: self.display_plagiarism_results(rt))
            except Exception as e:
                error_str = str(e)
                self.root.after(0, lambda err=error_str: self.display_plagiarism_error(err))
        
        thread = threading.Thread(target=check, daemon=True)
        thread.start()
    
    def format_plagiarism_results(self, per_reference, max_tfidf, avg_coverage, 
                                  max_coverage, estimated, ai_flags, ppl_thresh,
                                  ai_flagged_count, ai_flagged_percent, total_sentences):
        """Format plagiarism check results for display."""
        result = "=" * 70 + "\n"
        result += " " * 15 + "PLAGIARISM CHECK RESULTS\n"
        result += "=" * 70 + "\n\n"
        
        result += "📊 OVERALL SUMMARY\n"
        result += "-" * 70 + "\n"
        result += f"Max TF-IDF Similarity: {max_tfidf:.3f}\n"
        result += f"Average Exact Coverage: {avg_coverage*100:.2f}%\n"
        result += f"Max Exact Coverage: {max_coverage*100:.2f}%\n"
        result += f"Estimated Plagiarism: {estimated*100:.2f}%\n\n"
        
        result += "📁 PER-REFERENCE DETAILS\n"
        result += "-" * 70 + "\n"
        for ref in per_reference:
            result += f"\n📄 File: {ref['filename']}\n"
            result += f"   TF-IDF Similarity: {ref['tfidf']:.3f}\n"
            result += f"   Exact Coverage: {ref['coverage']*100:.2f}%\n"
            if ref['snippets']:
                result += f"   Matching Snippets ({len(ref['snippets'])} found):\n"
                for i, snippet in enumerate(ref['snippets'], 1):
                    result += f"   {i}. {snippet['context'][:120]}...\n"
                    result += f"      Similarity: {snippet['similarity']:.3f}\n"
            else:
                result += f"   No matching snippets found above threshold.\n"
            result += "\n"
        
        result += "=" * 70 + "\n"
        result += "🤖 AI-ORIGIN DETECTION\n"
        result += "-" * 70 + "\n"
        result += f"Perplexity Threshold: {ppl_thresh}\n"
        result += f"AI-likely Sentences: {ai_flagged_percent*100:.2f}% ({ai_flagged_count} of {total_sentences})\n\n"
        
        if ai_flags:
            for flag in ai_flags:
                status = "⚠️ [AI-LIKELY]" if flag['flagged'] else "✅ [HUMAN-LIKELY]"
                result += f"{status} (Perplexity: {flag['perplexity']:.2f})\n"
                result += f"   {flag['sentence']}\n\n"
        else:
            result += "No sentences found for analysis.\n"
        
        return result
    
    def display_plagiarism_results(self, result_text):
        """Display plagiarism check results."""
        self.plagiarism_results.config(state=tk.NORMAL)
        self.plagiarism_results.delete("1.0", tk.END)
        self.plagiarism_results.insert("1.0", result_text)
        self.plagiarism_results.config(state=tk.DISABLED)
    
    def display_plagiarism_error(self, error_msg):
        """Display plagiarism check error."""
        self.plagiarism_results.config(state=tk.NORMAL)
        self.plagiarism_results.delete("1.0", tk.END)
        self.plagiarism_results.insert("1.0", f"❌ Error: {error_msg}\n\nPlease check your input and try again.")
        self.plagiarism_results.config(state=tk.DISABLED)
        messagebox.showerror("Error", error_msg)

# ---------------------------
# Main Entry Point
# ---------------------------
def main():
    root = tk.Tk()
    app = TextIntelligenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

