import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module        

    def forward(self, input_seq, *args, **kwargs):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        size = list(input_seq.size())
        batch_size = size[0]
        time_steps = size[1]
        
        size_reshape = [batch_size * time_steps] + list(size[2:])
        
        reshaped_input = input_seq.contiguous().view(size_reshape)
        # Pass the additional arguments to the module
        output = self.module(reshaped_input, *args, **kwargs)

        # reshape output data to original shape
        output_size = [batch_size, time_steps] + list(output.size())[1:]        
        output = output.contiguous().view(output_size)

        return output

class MultiHeadAttention3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, kernel_size=3, stride=1, padding=1):
        super(MultiHeadAttention3D, self).__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        
        self.query_conv = nn.Conv3d(in_channels, out_channels * num_heads, kernel_size, stride, padding)
        self.key_conv = nn.Conv3d(in_channels, out_channels * num_heads, kernel_size, stride, padding)
        self.value_conv = nn.Conv3d(in_channels, out_channels * num_heads, kernel_size, stride, padding)
        
        self.final_conv = nn.Conv3d(out_channels * num_heads, out_channels, kernel_size, stride, padding)
        
    def forward(self, queries, keys, values):
        batch_size, time, channels, depth, height, width = queries.size()
        
        # Flatten the time and batch dimensions together for convolution
        queries = queries.view(batch_size * time, channels, depth, height, width)
        keys = keys.view(batch_size * time, channels, depth, height, width)
        values = values.view(batch_size * time, channels, depth, height, width)
        
        # Project the inputs to queries, keys, and values
        queries = self.query_conv(queries).view(batch_size, time, self.num_heads, self.out_channels, -1)
        keys = self.key_conv(keys).view(batch_size, time, self.num_heads, self.out_channels, -1)
        values = self.value_conv(values).view(batch_size, time, self.num_heads, self.out_channels, -1)
        
        queries = queries.permute(0, 2, 1, 4, 3)  # [batch_size, num_heads, time, depth*height*width, out_channels]
        keys = keys.permute(0, 2, 1, 4, 3)  # [batch_size, num_heads, time, depth*height*width, out_channels]
        values = values.permute(0, 2, 1, 4, 3)  # [batch_size, num_heads, time, depth*height*width, out_channels]
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.out_channels ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum of values
        attn_output = torch.matmul(attention_weights, values)
        attn_output = attn_output.permute(0, 2, 1, 4, 3).contiguous().view(batch_size * time, self.num_heads * self.out_channels, depth, height, width)
        
        # Final projection
        attn_output = self.final_conv(attn_output)
        
        # Reshape back to include time dimension
        attn_output = attn_output.view(batch_size, time, self.out_channels, depth, height, width)
        
        return attn_output, attention_weights


class ScoreLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, query):
        score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))
        return score
    

class ResnetBlock(nn.Module):
    def __init__(self, features, conv3d=False):
        super().__init__()
        self.features = features
        layers = []
        for i in range(2):
            layers += [
                nn.ReflectionPad2d(1) if not conv3d else nn.ReflectionPad3d(1),
                nn.Conv2d(features, features, kernel_size=3) if not conv3d else nn.Conv3d(features, features, kernel_size=3),
                nn.InstanceNorm2d(features) if not conv3d else nn.InstanceNorm3d(features),
            ]
            if i==0:
                layers += [
                    nn.ReLU(True)
                ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return input + self.model(input)
    
class Downsample(nn.Module):
    def __init__(self, features, conv3d=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1) if not conv3d else nn.ReflectionPad3d(1),
            nn.Conv2d(features, features, kernel_size=3, stride=2) if not conv3d else nn.Conv3d(features, features, kernel_size=3, stride=2),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)
    
class Upsample(nn.Module):
    def __init__(self, features, conv3d=False):
        super().__init__()
        layers = [
            nn.ReplicationPad2d(1) if not conv3d else nn.ReplicationPad3d(1),
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=3) if not conv3d else nn.ConvTranspose3d(features, features, kernel_size=4, stride=2, padding=3),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

class Head(nn.Module):
    def __init__(self, in_channels=1, features=64, residuals=9):
        super().__init__()

        mlp = nn.Sequential(*[
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ])
        mlp_id = 0
        setattr(self, 'mlp_%d' % mlp_id, mlp)
        mlp = nn.Sequential(*[
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ])
        mlp_id = 1
        setattr(self, 'mlp_%d' % mlp_id, mlp)
        for mlp_id in range(2, 5):
            mlp = nn.Sequential(*[
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ])
            setattr(self, 'mlp_%d' % mlp_id, mlp)

    def forward(self, feats):
        # if not encode_only:
        #     return(self.model(input))
        # else:
        #     num_patches = 256
        #     return_ids = []
        return_feats = []
        #     feat = input
        #     mlp_id = 0
        for feat_id, feat in enumerate(feats):
            mlp = getattr(self, 'mlp_%d' % feat_id)
            feat = mlp(feat)
            norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
            feat = feat.div(norm + 1e-7)
            return_feats.append(feat)
        return return_feats


class MLPHeads(nn.Module):
    def __init__(self, features=[64, 128, 256, 256, 256]):
        super().__init__()

        for mlp_id, feature in enumerate(features):
            
            mlp = nn.Sequential(*[
                nn.Linear(feature, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ])

            setattr(self, 'mlp_%d' % mlp_id, mlp)
        
    def forward(self, feats):
        
        return_feats = []
        
        for feat_id, feat in enumerate(feats):
            mlp = getattr(self, 'mlp_%d' % feat_id)
            feat = mlp(feat)
            norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
            feat = feat.div(norm + 1e-7)
            return_feats.append(feat)
        return return_feats

class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        self.beta_embed = nn.Embedding(num_classes, num_features)
        # Initialize scale close to 1 and bias as 0.
        nn.init.constant_(self.gamma_embed.weight, 1)
        nn.init.constant_(self.beta_embed.weight, 0)
    
    def forward(self, x, labels):
        out = self.instance_norm(x)
        gamma = self.gamma_embed(labels).unsqueeze(2).unsqueeze(3)
        beta = self.beta_embed(labels).unsqueeze(2).unsqueeze(3)
        return gamma * out + beta

class FiLM(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.features = in_features
        self.num_classes = num_classes
        self.film_gen = nn.Linear(num_classes, in_features * 2)  # Produces [gamma, beta]
    
    def forward(self, x, labels):
        labels = F.one_hot(labels, num_classes=self.num_classes).float()
        film_params = self.film_gen(labels)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

class ConditionalResnetBlock(nn.Module):
    def __init__(self, features, num_classes):
        super().__init__()
        layers = []
        self.features = features
        for i in range(2):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(features, features, kernel_size=3),
                ConditionalInstanceNorm2d(features, num_classes)  
            ]
            if i==0:
                layers += [
                    nn.ReLU(True)
                ]
        self.model = nn.Sequential(*layers)

    def forward(self, input, labels):
        for layer in self.model:
            if hasattr(layer, 'forward') and 'labels' in layer.forward.__code__.co_varnames[:layer.forward.__code__.co_argcount]:
                input = layer(input, labels)
            else:
                input = layer(input)
        return input

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class ProjectionHead(nn.Module):
    # Projection MLP
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128, dropout=0.1, activation=nn.ReLU, bias=False):   
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=bias),
            nn.Identity() if activation is None else activation(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_dim, bias=bias),
        )

    def forward(self, x, x_=None):
        if x_ is not None:
            x = torch.cat([x, x_], dim=-1)
        x = self.model(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dim=1):
        super().__init__()

        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        self.dim = dim

    def forward(self, query, values):
        
        score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))

        attention_weights = score/torch.sum(score, dim=self.dim, keepdim=True)

        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=self.dim)

        return context_vector, score
    
class AttentionChunk(nn.Module):
    def __init__(self, input_dim, hidden_dim, chunks=16, time_dim=1):
        super().__init__()
        
        self.attn = SelfAttention(input_dim, hidden_dim, dim=time_dim)
        self.chunks = chunks
        self.time_dim = time_dim

    def forward(self, x):
        
        x_out = []
        x_s = []
        
        ch_idx = 0
        for ch in torch.tensor_split(x, self.chunks, dim=self.time_dim): # Iterate in the time dimension and create chunks            
            ch, ch_s = self.attn(ch, ch) # Compute average attention for each chunk            
            x_out.append(ch)
            x_s.append(ch_s)
            ch_idx += 1
        x_out = torch.stack(x_out, dim=self.time_dim)
        x_s = torch.cat(x_s, dim=self.time_dim)

        return x_out, x_s

class ContextModulated(nn.Module):
    def __init__(self, input_dim, output_dim, context_dim, activation=nn.LeakyReLU):
        super(ContextModulated, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        
        self.hyper_gate = nn.Linear(context_dim, output_dim)
        self.hyper_bias = nn.Linear(context_dim, output_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = nn.Identity()

    def forward(self, x, context):
        
        
        gate = self.sigmoid(self.hyper_gate(context))
        bias = self.hyper_bias(context)        
        
        return self.activation(self.fc(x)*gate + bias)  
    
class MHAContextModulated(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim, dropout=0.1, causal_mask=False, return_weights=False):
        super(MHAContextModulated, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal_mask = causal_mask
        self.return_weights = return_weights        

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False, batch_first=True)
        self.context_modulated = ContextModulated(input_dim=embed_dim, output_dim=output_dim, context_dim=embed_dim)
    
    def forward(self, x):

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        context, attn_output_weights = self.attention(q, k, v, is_causal=self.causal_mask)
        x = self.context_modulated(x, context)
        
        if self.return_weights:
            return x, attn_output_weights
        return x
    
class MHABlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, causal_mask=False, return_weights=False):
        super(MHABlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal_mask = causal_mask
        self.return_weights = return_weights

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False, batch_first=True)

    def generate_causal_mask(self, seq_len):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    
    def forward(self, x):

        mask = None
        if self.causal_mask:
            mask = self.generate_causal_mask(x.size(1)).to(x.device) 

        x, attn_output_weights = self.attention(x, x, x, attn_mask=mask, is_causal=self.causal_mask)

        if self.return_weights:
            return x, attn_output_weights
        return x

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )
        self.proj_q = nn.Linear(dim_in, dim_out)
        self.proj_k = nn.Linear(dim_in, dim_out)
        self.proj_v = nn.Linear(dim_in, dim_out)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)

    def forward(self, Q, K):
        Q_proj = self.proj_q(Q)
        K_proj = self.proj_k(K)
        V_proj = self.proj_v(K)

        out, _ = self.attn(Q_proj, K_proj, V_proj)
        out = self.ln1(out + Q_proj)
        out = self.ln2(out + self.fc(out))
        return out

class InducedSetAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_induce):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_induce, dim_out))
        self.mab1 = MultiheadAttentionBlock(dim_in, dim_out, num_heads)
        self.mab2 = MultiheadAttentionBlock(dim_out, dim_out, num_heads)

    def forward(self, X):
        B = X.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X)
        return self.mab2(X, H)

class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, num_seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, input_dim))
        self.mab = MultiheadAttentionBlock(input_dim, input_dim, num_heads)

    def forward(self, X):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X)  # [B, num_seeds, D]

class SetTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_inds=16, num_seeds=1, num_isabs=2):
        super().__init__()
        layers = []
        for _ in range(num_isabs):
            layers.append(InducedSetAttentionBlock(input_dim if _ == 0 else hidden_dim, hidden_dim, num_heads, num_inds))
        self.encoder = nn.Sequential(*layers)
        self.pma = PoolingByMultiheadAttention(hidden_dim, num_heads, num_seeds)

    def forward(self, X):
        # X: [B, N, D_in]
        out = self.encoder(X)
        pooled = self.pma(out)  # [B, num_seeds, D]
        return pooled.squeeze(1)  # [B, D]

class OrientationPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.set_encoder = SetTransformerEncoder(input_dim=input_dim)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # 3x3 matrix
        )

    def forward(self, x):
        # x: [B, N, D]
        z = self.set_encoder(x)         # [B, D]
        R = self.fc(z).view(-1, 3, 3)   # [B, 3, 3]
        return R