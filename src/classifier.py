"""TaxoClass classifier (paper Sect. 3.3.1): dual-encoder + matching network.

- Document encoder: BERT-base-uncased, [CLS] token representation.
- Class encoder: GCN over the taxonomy (Eq. 4-5). Implementation choices,
  documented for the report:
    * The paper propagates over each class's ego network with shared weights;
      we run the (equivalent-in-spirit) shared-weight GCN over the full
      taxonomy graph and then read out each class as the MEAN over its ego
      network {c} ∪ Par(c) ∪ Chd(c)  (Eq. 5).
    * Node features are initialized from the document encoder's own input
      word embeddings averaged over class-name tokens (the paper uses
      pre-trained word embeddings; footnote 3 averages multi-gram names).
      Features are kept fixed; the GCN weights W^(l) are learned.
- Matching network: Eq. 6 as printed is p = sigmoid(exp(c^T B d)), which is
  bounded in (0.5, 1) and makes the BCE negative term ill-posed - almost
  certainly a typo. We use the standard bilinear form p = sigmoid(c^T B d).
"""
import numpy as np
import torch
import torch.nn as nn


def build_graph_tensors(taxo, device):
    """Symmetric-normalized adjacency with self-loops (Eq. 4's alpha_uv)
    and the ego-network mean-readout matrix (Eq. 5)."""
    C = len(taxo.names)
    A = np.zeros((C, C), dtype=np.float32)
    for c, ps in taxo.parents.items():
        for p in ps:
            A[c, p] = A[p, c] = 1.0
    np.fill_diagonal(A, 1.0)                      # N(u) includes u itself
    deg = A.sum(1)
    A = A / np.sqrt(np.outer(deg, deg))           # alpha_uv = 1/sqrt(|N(u)||N(v)|)

    M = np.zeros((C, C), dtype=np.float32)        # ego mean readout
    for j in range(C):
        ego = {j} | set(taxo.parents.get(j, ())) | set(taxo.children.get(j, ()))
        for u in ego:
            M[j, u] = 1.0 / len(ego)
    return (torch.tensor(A, device=device), torch.tensor(M, device=device))


def init_class_features(taxo, tokenizer, embedding_layer):
    """Average the doc encoder's input word embeddings over class-name tokens."""
    C = len(taxo.names)
    H = embedding_layer.weight.shape[1]
    feats = torch.zeros(C, H)
    with torch.no_grad():
        for c in range(C):
            ids = tokenizer(taxo.names[c], add_special_tokens=False)["input_ids"]
            if ids:
                feats[c] = embedding_layer.weight[ids].mean(0)
    # BERT input embeddings are anisotropic (all vectors share a dominant
    # common direction), making class features near-collinear and the
    # matching logits nearly class-independent. Center across classes to
    # remove the common component, then L2-normalize rows for logit scale.
    feats = feats - feats.mean(dim=0, keepdim=True)
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
    return feats


class TaxoClassifier(nn.Module):
    def __init__(self, bert, taxo, tokenizer, gnn_layers=2, device="cpu"):
        super().__init__()
        self.bert = bert
        H = bert.config.hidden_size
        self.adj, self.ego = build_graph_tensors(taxo, device)
        feats = init_class_features(taxo, tokenizer, bert.get_input_embeddings())
        # learnable per-class identity (cf. the paper's NoGNN ablation which
        # uses a learnable embedding layer); a fixed buffer leaves no path
        # for classes to become separable when initialization is collinear
        self.node_feat = nn.Parameter(feats)
        self.gnn = nn.ModuleList([nn.Linear(H, H, bias=False)
                                  for _ in range(gnn_layers)])
        self.B = nn.Parameter(torch.eye(H) + 0.01 * torch.randn(H, H))

    def class_reps(self):
        h = self.node_feat
        for W in self.gnn:                        # Eq. 4, + residual: without
            h = h + torch.relu(self.adj @ W(h))   # it, 2 hops + ego-mean
        return self.ego @ h                       # over-smooth siblings. Eq. 5

    def forward(self, input_ids, attention_mask):
        D = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state[:, 0]
        Creps = self.class_reps()
        return (D @ self.B) @ Creps.T             # logits; sigmoid -> Eq. 6 (fixed)

    def bert_parameters(self):
        return self.bert.parameters()

    def head_parameters(self):
        yield self.node_feat
        for m in (self.gnn, [self.B]):
            for p in (m.parameters() if isinstance(m, nn.ModuleList) else m):
                yield p
