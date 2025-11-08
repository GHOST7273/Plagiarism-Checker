import os
import io
import math
from flask import Flask, request, render_template_string, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
import difflib
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ensure punkt is available
nltk.download('punkt', quiet=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------
# Utilities: text extraction
# ---------------------------
def load_text_from_file(filepath):
    if filepath.lower().endswith('.pdf'):
        return extract_text(filepath)
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

# ---------------------------
# Plagiarism detection (TF-IDF + snippet matching)
# ---------------------------
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
# Load once (may be slow on first run)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt2_model_name = "gpt2"  # you can use "gpt2-medium" if you have memory/GPU
tokenizer_gpt2 = AutoTokenizer.from_pretrained(gpt2_model_name)
model_gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(device)
model_gpt2.eval()

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
# Flask routes
# ---------------------------
INDEX_HTML = """
<!doctype html>
<title>Plagiarism + AI-checker</title>
<h2>Plagiarism & AI-origin Checker</h2>
<form method=post enctype=multipart/form-data>
  <label>Paste text to check (required):</label><br>
  <textarea name="query" rows="12" cols="100">{{ request_form.get('query','') }}</textarea><br><br>

  <label>Upload reference files (txt or pdf). You can upload several:</label><br>
  <input type=file name=refs multiple><br><br>

  <label>TF-IDF similarity threshold (0-1):</label>
  <input type=text name="tfidf_thresh" value="{{ request_form.get('tfidf_thresh', '0.4') }}"><br>

  <label>AI perplexity threshold (lower -> more likely AI):</label>
  <input type=text name="ppl_thresh" value="{{ request_form.get('ppl_thresh','40.0') }}"><br><br>

  <input type=submit value="Check">
</form>
<hr>
{% if results %}
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
  <ul>
    {% for s in results.ai_flags %}
      <li>
        {% if s.flagged %}<strong style="color:red">[AI-likely]</strong>{% else %}<strong style="color:green">[Human-likely]</strong>{% endif %}
        Perplexity: {{ s.perplexity|round(2) }} — {{ s.sentence }}
      </li>
    {% endfor %}
  </ul>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = (request.form.get('query') or "").strip()
        if not query:
            return render_template_string(INDEX_HTML, request_form=request.form, results=None, error="Query empty")

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

        results = {
            'per_reference': per_reference,
            'max_tfidf': max_tfidf,
            'estimated_plagiarism': estimated,
            'ai_flags': ai_flags,
            'ppl_thresh': ppl_thresh
        }

        return render_template_string(INDEX_HTML, request_form=request.form, results=results)

    return render_template_string(INDEX_HTML, request_form={}, results=None)

if __name__ == '__main__':
    app.run(debug=True)
