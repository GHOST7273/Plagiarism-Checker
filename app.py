import os
import io
import math
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, url_for, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
import difflib
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

# Ensure NLTK tokenizer resources are available (handles newer punkt_tab split)
for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        if _resource == "punkt":
            raise
        # Older NLTK versions may not expose punkt_tab; ignore if download fails

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------
# Load Models
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paraphrasing model
print("Loading paraphrasing model...")
paraphrase_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# GPT-2 for AI detection
print("Loading GPT-2 model for AI detection...")
gpt2_model_name = "gpt2"
tokenizer_gpt2 = AutoTokenizer.from_pretrained(gpt2_model_name)
model_gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(device)
model_gpt2.eval()

# ---------------------------
# Paraphrasing Functions
# ---------------------------
def paraphrase_text(sentence):
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

# ---------------------------
# Plagiarism Detection Functions
# ---------------------------
def load_text_from_file(filepath):
    if filepath.lower().endswith('.pdf'):
        return extract_text(filepath)
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def compute_tfidf_similarities(query_text, reference_texts):
    """
    returns list of (source_index, cosine_sim) where higher is more similar
    """
    corpus = [query_text] + reference_texts
    vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
    tfidf = vectorizer.fit_transform(corpus)
    query_vec = tfidf[0]
    refs = tfidf[1:]
    sims = cosine_similarity(query_vec, refs)[0]
    return sims  # numpy array

def get_matching_snippets(query, reference, cutoff=0.6, max_snippets=5):
    """
    Simple heuristic: use difflib to find matching blocks and return snippets
    cutoff = minimum ratio for a matching block
    """
    matcher = difflib.SequenceMatcher(None, query, reference)
    blocks = matcher.get_matching_blocks()
    snippets = []
    for b in reversed(blocks):  # reversed: longer matches usually near end; we'll filter
        i, j, size = b
        if size <= 20:  # ignore tiny matches (tunable)
            continue
        match = query[i:i+size].strip()
        ratio = matcher.ratio()
        # record snippet with context
        start = max(0, i-40)
        end = min(len(query), i+size+40)
        context = query[start:end].replace('\n', ' ')
        snippets.append({
            'match': match,
            'context': context,
            'match_length': size
        })
        if len(snippets) >= max_snippets:
            break
    return snippets

def compute_exact_coverage(query, reference):
    """
    Very simple: slide over query in windows and check for substring presence.
    Returns fraction of characters (approx) of query that appear verbatim in reference.
    """
    q = query
    ref = reference
    total = 0
    matched = 0
    window = 50  # characters; tune as needed
    i = 0
    while i < len(q):
        total += min(window, len(q) - i)
        segment = q[i:i+window]
        if segment.strip() and segment in ref:
            matched += len(segment)
            i += window  # skip ahead if matched
        else:
            i += int(window/4)  # step smaller to find overlaps
    if total == 0:
        return 0.0
    return matched / total

# ---------------------------
# AI-detection using GPT-2 perplexity heuristic
# ---------------------------
def perplexity_of_text(text):
    """
    Compute approximate perplexity of a piece of text using GPT-2.
    Returns float (lower -> more likely under GPT-2).
    """
    enc = tokenizer_gpt2(text, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(device)
    with torch.no_grad():
        outputs = model_gpt2(input_ids, labels=input_ids)
        # loss is the average cross-entropy per token
        loss = outputs.loss.item()
    ppl = math.exp(loss) if loss < 50 else float('inf')
    return ppl

def ai_sentence_flags(text, threshold=40.0):
    """
    Tokenize into sentences and compute perplexity per sentence.
    Sentences with perplexity below threshold are flagged as likely AI-generated.
    Returns list of dicts: {sentence, perplexity, flagged}
    """
    sents = sent_tokenize(text)
    results = []
    for s in sents:
        p = perplexity_of_text(s)
        flagged = p < threshold
        results.append({'sentence': s, 'perplexity': p, 'flagged': flagged})
    return results

# ---------------------------
# HTML Templates
# ---------------------------
NAVIGATION_HTML = """
<style>
    body {
        font-family: Arial, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
    }
    .nav-container {
        background-color: #fff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .logo-area {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 15px;
    }
    .logo-area img {
        height: 60px;
        width: auto;
        border-radius: 8px;
    }
    .logo-text {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .logo-text .brand-name {
        font-size: 20px;
        font-weight: bold;
        color: #222;
        letter-spacing: 0.5px;
    }
    .logo-text .tagline {
        font-size: 13px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .nav-buttons {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
    }
    .nav-btn {
        padding: 12px 24px;
        background-color: #4CAF50;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-size: 16px;
        transition: background-color 0.3s;
        flex: 1 1 200px;
        text-align: center;
    }
    .nav-btn:hover {
        background-color: #45a049;
    }
    .nav-btn.active {
        background-color: #2196F3;
    }
    .content-container {
        background-color: #fff;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #333;
        margin-top: 0;
    }
    h2 {
        color: #555;
    }
    textarea, input[type="text"], input[type="file"] {
        width: 100%;
        box-sizing: border-box;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 14px;
        font-family: Arial, sans-serif;
    }
    input[type="submit"] {
        padding: 12px 20px;
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
        margin-top: 10px;
        width: 100%;
    }
    input[type="submit"]:hover {
        background-color: #0b7dda;
    }
    label {
        display: block;
        margin-top: 15px;
        font-weight: bold;
        color: #555;
    }
    .result-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
        border-left: 4px solid #2196F3;
    }
    ul {
        line-height: 1.6;
        padding-left: 20px;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: monospace;
    }
    .ai-summary {
        margin-bottom: 15px;
        padding: 10px;
        background-color: #eef6ff;
        border-radius: 4px;
        border-left: 4px solid #2196F3;
    }
    .ai-summary strong {
        color: #1a73e8;
    }
    .site-footer {
        margin-top: 40px;
        text-align: center;
        color: #777;
        font-size: 14px;
        padding: 20px 10px;
    }
    .site-footer a {
        color: #2196F3;
        text-decoration: none;
    }
    .site-footer a:hover {
        text-decoration: underline;
    }
    @media (max-width: 768px) {
        body {
            padding: 15px;
        }
        .nav-buttons {
            gap: 10px;
        }
        .content-container {
            padding: 20px;
        }
        textarea {
            min-height: 180px;
        }
    }
    @media (max-width: 540px) {
        h1 {
            font-size: 24px;
        }
        h2 {
            font-size: 20px;
        }
        .logo-area {
            flex-direction: column;
            align-items: flex-start;
        }
        .nav-buttons {
            flex-direction: column;
        }
        .nav-btn {
            flex: 1 1 auto;
        }
        input[type="submit"] {
            font-size: 15px;
        }
    }
</style>
<div class="nav-container">
    <div class="logo-area">
        <img src="{{ url_for('ghost_logo') }}" alt="GHOST logo">
        <div class="logo-text">
            <span class="brand-name">GHOST LLP</span>
            <span class="tagline">Text Intelligence Suite</span>
        </div>
    </div>
    <div class="nav-buttons">
        <a href="/" class="nav-btn {{ 'active' if active_page == 'home' else '' }}">Home</a>
        <a href="/paraphrase" class="nav-btn {{ 'active' if active_page == 'paraphrase' else '' }}">Paraphrasing Tool</a>
        <a href="/plagiarism" class="nav-btn {{ 'active' if active_page == 'plagiarism' else '' }}">Plagiarism Checker</a>
    </div>
</div>
"""

HOME_HTML = NAVIGATION_HTML + """
<div class="content-container">
    <h1>Welcome to Text Processing Tools</h1>
    <p>Choose a tool from the navigation menu above:</p>
    <ul>
        <li><strong>Paraphrasing Tool:</strong> Rewrite your text to create unique variations while maintaining the original meaning.</li>
        <li><strong>Plagiarism Checker:</strong> Check your text for plagiarism against reference documents and detect AI-generated content.</li>
    </ul>
</div>
"""

FOOTER_HTML = """
<footer class="site-footer">
    © {{ current_year }} GHOST LLP · Crafted with care for smarter writing assistance.
</footer>
"""

PARAPHRASE_HTML = NAVIGATION_HTML + """
<div class="content-container">
    <h1>Paraphrasing Tool</h1>
    <form method="post">
        <label for="text">Enter text to paraphrase:</label>
        <textarea name="text" id="text" rows="10" cols="80" placeholder="Enter your text here...">{{ text if text else '' }}</textarea><br><br>
        <input type="submit" value="Paraphrase">
    </form>
    {% if result %}
    <div class="result-box">
        <h3>Paraphrased Text:</h3>
        <p>{{ result }}</p>
    </div>
    {% endif %}
</div>
""" + FOOTER_HTML

PLAGIARISM_HTML = NAVIGATION_HTML + """
<div class="content-container">
    <h1>Plagiarism & AI-origin Checker</h1>
    <form method="post" enctype="multipart/form-data">
        <label>Paste text to check (required):</label>
        <textarea name="query" rows="12" cols="100" placeholder="Enter text to check for plagiarism...">{{ request_form.get('query','') }}</textarea><br><br>

        <label>Upload reference files (txt or pdf). You can upload several:</label>
        <input type="file" name="refs" multiple><br><br>

        <label>TF-IDF similarity threshold (0-1):</label>
        <input type="text" name="tfidf_thresh" value="{{ request_form.get('tfidf_thresh', '0.4') }}"><br>

        <label>AI perplexity threshold (lower -> more likely AI):</label>
        <input type="text" name="ppl_thresh" value="{{ request_form.get('ppl_thresh','40.0') }}"><br><br>

        <input type="submit" value="Check">
    </form>
    <hr>
    {% if results %}
    <div class="result-box">
        <h3>Overall Plagiarism Summary</h3>
        <p><strong>Max TF-IDF similarity against any reference:</strong> {{ results.max_tfidf|round(3) }}</p>
        <p><strong>Estimated plagiarism percent (heuristic):</strong> {{ (results.estimated_plagiarism*100)|round(2) }}%</p>

        <h3>Per-reference details</h3>
        <ul>
        {% for r in results.per_reference %}
            <li>
                <strong>{{ r.filename }}</strong> — TF-IDF sim: {{ r.tfidf|round(3) }}, exact coverage: {{ (r.coverage*100)|round(2) }}%
                <ul>
                    <li>Top matching snippets:
                        <ul>
                            {% for s in r.snippets %}
                                <li><code>{{ s.context }}</code></li>
                            {% endfor %}
                        </ul>
                    </li>
                </ul>
            </li>
        {% endfor %}
        </ul>

        <h3>AI-origin detection (per-sentence)</h3>
        <p>Perplexity threshold used: {{ results.ppl_thresh }}</p>
        <div class="ai-summary">
            {% if results.ai_total_sentences > 0 %}
                <strong>AI-likely sentences:</strong> {{ (results.ai_flagged_percent*100)|round(2) }}% ({{ results.ai_flagged_count }} of {{ results.ai_total_sentences }})
            {% else %}
                <strong>AI-likely sentences:</strong> No sentences available for analysis.
            {% endif %}
        </div>
        <ul>
            {% for s in results.ai_flags %}
                <li>
                    {% if s.flagged %}<strong style="color:red">[AI-likely]</strong>{% else %}<strong style="color:green">[Human-likely]</strong>{% endif %}
                    Perplexity: {{ s.perplexity|round(2) }} — {{ s.sentence }}
                </li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
</div>
""" + FOOTER_HTML

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/ghostlogo.webp')
def ghost_logo():
    return send_file('ghostlogo.webp', mimetype='image/webp')

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HOME_HTML, active_page='home', current_year=datetime.now().year)

@app.route('/paraphrase', methods=['GET', 'POST'])
def paraphrase():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if not text:
            return render_template_string(PARAPHRASE_HTML, active_page='paraphrase', text='', result=None, error="Please enter some text to paraphrase", current_year=datetime.now().year)
        try:
            result = paraphrase_text(text)
            return render_template_string(PARAPHRASE_HTML, active_page='paraphrase', text=text, result=result, current_year=datetime.now().year)
        except Exception as e:
            return render_template_string(PARAPHRASE_HTML, active_page='paraphrase', text=text, result=None, error=f"Error: {str(e)}", current_year=datetime.now().year)
    
    return render_template_string(PARAPHRASE_HTML, active_page='paraphrase', text='', result=None, current_year=datetime.now().year)

@app.route('/plagiarism', methods=['GET', 'POST'])
def plagiarism_check():
    if request.method == 'POST':
        query = (request.form.get('query') or "").strip()
        if not query:
            return render_template_string(PLAGIARISM_HTML, active_page='plagiarism', request_form=request.form, results=None, error="Query empty", current_year=datetime.now().year)

        # save uploaded refs
        refs = []
        for f in request.files.getlist('refs'):
            if f and f.filename:
                path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                f.save(path)
                refs.append({'filename': f.filename, 'path': path})

        # load reference texts
        reference_texts = []
        for r in refs:
            txt = load_text_from_file(r['path'])
            reference_texts.append(txt)

        # TF-IDF similarities
        tfidf_thresh = float(request.form.get('tfidf_thresh', 0.4))
        sims = compute_tfidf_similarities(query, reference_texts) if reference_texts else []
        per_reference = []
        max_tfidf = 0.0
        total_coverage = 0.0
        for idx, r in enumerate(refs):
            tfidf_score = float(sims[idx]) if len(sims) > idx else 0.0
            if tfidf_score > max_tfidf:
                max_tfidf = tfidf_score
            coverage = compute_exact_coverage(query, reference_texts[idx])
            total_coverage = max(total_coverage, coverage)
            snippets = get_matching_snippets(query, reference_texts[idx], cutoff=tfidf_thresh)
            per_reference.append({
                'filename': r['filename'],
                'tfidf': tfidf_score,
                'coverage': coverage,
                'snippets': snippets
            })

        # Estimate plagiarism percentage (heuristic combination)
        # Weighted sum of max TF-IDF similarity (0-1) and exact coverage (0-1)
        estimated = 0.6 * max_tfidf + 0.4 * total_coverage

        # AI detection
        ppl_thresh = float(request.form.get('ppl_thresh', 40.0))
        ai_flags = ai_sentence_flags(query, threshold=ppl_thresh)
        total_sentences = len(ai_flags)
        ai_flagged_count = sum(1 for item in ai_flags if item['flagged'])
        ai_flagged_percent = (ai_flagged_count / total_sentences) if total_sentences else 0.0

        results = {
            'per_reference': per_reference,
            'max_tfidf': max_tfidf,
            'estimated_plagiarism': estimated,
            'ai_flags': ai_flags,
            'ppl_thresh': ppl_thresh,
            'ai_total_sentences': total_sentences,
            'ai_flagged_count': ai_flagged_count,
            'ai_flagged_percent': ai_flagged_percent
        }

        return render_template_string(PLAGIARISM_HTML, active_page='plagiarism', request_form=request.form, results=results, current_year=datetime.now().year)

    return render_template_string(PLAGIARISM_HTML, active_page='plagiarism', request_form={}, results=None, current_year=datetime.now().year)

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
