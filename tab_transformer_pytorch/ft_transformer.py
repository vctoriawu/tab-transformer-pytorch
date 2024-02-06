import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from .embeddings import NEmbedding

# feedforward and attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * 2), # * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim) # * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 4,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        hidden_dim = 128,
        numerical_features: list = None,
        classification: bool = False,
        emb_type = "ple",
        mask_prob=0.15,
        numerical_bins=10
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NEmbedding(self.num_continuous, numerical_features, emb_dim=dim, emb_type=emb_type, mask_prob=mask_prob, n_bins=numerical_bins)

        # total number of features
        
        self.num_features = len(categories) + num_continuous
        
        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))


        # Transformers

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

        self.is_classification = classification

        # to reconstruction

        self.embedding_dim = dim
        self.to_reconstruction = nn.Sequential(
            nn.LayerNorm(self.num_features * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.num_features * self.embedding_dim, hidden_dim),  # First hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_features) 
        )

        parameters = {
            'categories': categories,
            'num_continuous': num_continuous,
            'dim': dim,
            'depth': depth,
            'heads': heads,
            'dim_head': dim_head,
            'dim_out': dim_out,
            'num_special_tokens': num_special_tokens,
            'attn_dropout': attn_dropout,
            'ff_dropout': ff_dropout,
            'hidden_dim': hidden_dim,
            'numerical_features': numerical_features,
            'classification': classification,
            'emb_type': emb_type,
            'numerical_bins': numerical_bins
        }

        torch.save(parameters, "model_parameters.pth")
    
    def mask_inputs(self, inputs):
        """
        Randomly mask input values during training
        """
        # Generate a mask for each column
        column_mask = torch.rand(inputs.size(1)) < self.mask_prob

        # Tile the column mask to match the shape of inputs
        column_mask = column_mask.unsqueeze(0).expand(inputs.size(0), -1)

        # Replace masked columns with MASK_VALUE
        masked_inputs = inputs.clone()
        masked_inputs[:, column_mask] = self.MASK_VALUE

        return masked_inputs, column_mask

    def forward(self, x_numer, x_categ = [], return_attn = False, mask = False):
        #assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            # Implement random masking during training
            if self.training:
                x_categ = self.mask_inputs(x_categ)

            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            # Numerical embedding has masking defined in the class
            x_numer = self.numerical_embedder(x_numer, mask).to(torch.float32)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        if self.is_classification:
            b = x.shape[0]
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        #for transformer_block in self.transformer:
        #    x, attns = transformer_block(x)

        x, attns = self.transformer(x, return_attn = True)

        if self.is_classification:
            # get cls token

            cls_x = x[:, 0]

            # out in the paper is linear(relu(ln(cls)))

            logits = self.to_logits(cls_x)

            if not return_attn:
                return logits, x

            return logits, x, attns
        
        else: # reconstruction
            reshaped_x = x.view(-1, self.num_features * self.embedding_dim)
            reconstructed_input = self.to_reconstruction(reshaped_x)

            if not return_attn:
                return reconstructed_input  # Return None for attention maps in reconstruction task

            return reconstructed_input, attns