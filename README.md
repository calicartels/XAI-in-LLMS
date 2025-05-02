# XAI-in-LLMS

## Overview

**XAI-in-LLMS** is a research/development repo for exploring explainability (XAI) in large language models (LLMs). The focus is on technical, reproducible experiments with SOTA transformer models, embedding models, and dimensionality reduction for interpretability. 

## Stack
- Python 3.10+ (conda recommended)
- PyTorch 2.1.x (tested)
- Transformers 4.30.x (tested)
- sentence-transformers 2.2.x
- scikit-learn, umap-learn, matplotlib, tqdm, plotly, pandas
- HuggingFace Hub (for model access)

## Technical Details

### Embedding Models
- **Nvidia NV-Embed-v1** ([nvidia/nv-embed-v1](https://huggingface.co/nvidia/nv-embed-v1))
- **LinkedIn Linq-Embed-Mistral** ([linkedin/linq-embed-mistral](https://huggingface.co/linkedin/linq-embed-mistral))
- **BAAI bge-large-en-v1.5** ([BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5))
- Fallback: `sentence-transformers/all-MiniLM-L6-v2` if above fail

### Workflow
1. **Text Data**: Uses a curated set of tech/AI/ML sentences (see notebook for details)
2. **Embeddings**: Generates dense vector representations using the above models via `sentence-transformers` API
3. **Dimensionality Reduction**:
   - **PCA** (linear, fast, loses nonlinear structure)
   - **t-SNE** (nonlinear, good for small datasets, preserves local structure, distorts global)
   - **UMAP** (nonlinear, scalable, preserves local/global, fast)
4. **Visualization**: 3D scatter plots via Plotly, with interactive hover for text inspection
5. **Reproducibility**: All random seeds set, all model/tokenizer versions pinned, fallback logic for model download failures
6. **Extensibility**: Add new models or reduction methods by extending the notebook functions

### Example Code Snippet
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('nvidia/nv-embed-v1', token=HF_TOKEN, trust_remote_code=True)
embeddings = model.encode(texts)
```

### Dimensionality Reduction Example
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
pca = PCA(n_components=3).fit_transform(embeddings)
tsne = TSNE(n_components=3, random_state=42, perplexity=5, n_iter=1000).fit_transform(embeddings)
umap_result = umap.UMAP(n_components=3, random_state=42, n_neighbors=5).fit_transform(embeddings)
```

### Visualization Example
```python
import plotly.express as px
df = pd.DataFrame(pca, columns=["x", "y", "z"])
df["text"] = texts
fig = px.scatter_3d(df, x="x", y="y", z="z", text="text")
fig.show()
```

## Setup

### 1. Clone and enter repo
```bash
git clone <this-repo-url>
cd XAI-in-LLMS
```

### 2. Create and activate environment (conda recommended)
```bash
conda create -n xai_env python=3.10 -y
conda activate xai_env
```

### 3. Install dependencies
```bash
pip install torch==2.1.0 torchvision==0.16.0
pip install transformers==4.30.2 sentence-transformers==2.2.2
pip install scikit-learn umap-learn matplotlib tqdm plotly pandas python-dotenv
```

## Usage

- Main code is in `XAI_in_LLMs.ipynb` (Jupyter notebook)
- Launch with:
```bash
jupyter notebook
```
- If you want to run scripts, adapt the notebook cells as needed.
- HuggingFace token required for some models: set `HF_TOKEN` in your environment or `.env` file

## Troubleshooting

- **Circular import / torch._dynamo errors:**
    - Use the exact versions above. Newer PyTorch/transformers combos are often broken.
    - If you see `torch._dynamo has no attribute 'disable'`, downgrade PyTorch and transformers as above.
- **Apple Silicon (M1/M2/M3):**
    - Use arm64 wheels. If you get weird errors, try `pip install --force-reinstall --no-cache-dir torch torchvision`.
- **CUDA:**
    - This setup is CPU by default. For GPU, install the right CUDA-enabled torch wheel for your hardware.
- **Model Download Fails:**
    - Check your HuggingFace token and internet connection. Fallback to MiniLM is automatic.

## Contribution/Dev
- PRs welcome if you know what you're doing.
- Use issues for technical discussion only.
- If you add new notebooks or scripts, update this README.
- Add new models or reduction methods by extending the notebook functions.

## References
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [NV-Embed-v1](https://huggingface.co/nvidia/nv-embed-v1)
- [Linq-Embed-Mistral](https://huggingface.co/linkedin/linq-embed-mistral)
- [BAAI bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

---

If you break it, you get to keep both pieces. Ping me for real research collabs.

