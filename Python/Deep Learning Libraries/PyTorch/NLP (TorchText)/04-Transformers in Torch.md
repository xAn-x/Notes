## Transformer [](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#transformer)

``` python
import torch
import torch.nn as nn
import torch.functional as F


# Create Instance of transformer
transformer=nn.Transformer(d_model=512,nhead=8,num_encoder_layers=6,num_decoder_layers=6
						  ,dim_feedforward=2048,dropout=0.1,activation=F.relu,
						  custom_encoder=None,custom_decoder=None)

# forward Pass
# causal -> causal mask to prevent attending to future tokens.
# memory_mask:
	# - control the attention mechanism in the decoder part of the transformer model.
	# - for cross-attention part
out=transformer(src,tgt,src_mask=None,tgt_mask=None,memory_mask=None
				src_key_padding_mask=None,tgt_key_padding_mask=None,
				memory_key_padding_mask=None,
				src_is_causal=None, tgt_is_causal=None, memory_is_causal=False)
```

## Encoder and Decoder

*TransformerEncoder[](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#transformerencoder)*

```python
# TransformerEncoder is a stack of N encoder layers.
import torch
import torch.nn as nn
import torch.functional as F

encoder=nn.TransformerEncoder(encoder_layer,num_layers,norm=None,
							  enable_nested_tensor=True,mask_check=True)

# encoder_layer (TransformerEncoderLayer) – an instance of the TransformerEncoderLayer() class (required).
# num_layers (int) – the number of sub-encoder-layers in the encoder (required).
# norm (Optional[Module]) – the layer normalization component (optional).
# enable_nested_tensor (bool) – if True, input will automatically convert to nested tensor (and convert back on output). This will improve the overall performance of TransformerEncoder when padding rate is high. Default: True (enabled).


# Forward pass
out=encoder(src,mask,src_key_padding_mask=None,is_causal=None)
```

*TransformerDecoder[](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html#transformerdecoder)*

```python
# TransformerEncoder is a stack of N encoder layers.
import torch
import torch.nn as nn
import torch.functional as F

encoder=nn.TransformerDecoder(decoder_layer,num_layers,norm=None)

# decoder_layer (TransformerEncoderLayer) – an instance of the TransformerEncoderLayer() class (required).
# num_layers (int) – the number of sub-encoder-layers in the encoder (required).
# norm (Optional[Module]) – the layer normalization component (optional).

# Forward pass
out=decoder(tgt,tgt_mask,memory_mask=None,
			tgt_key_padding_mask=None,memory_key_padding_mask=None,
			tgt_is_causal=None,memory_is_causal=False)

```

*TransformerEncoderLayer[](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#transformerencoderlayer)*

```python
import torch
import torch.nn as nn
import torch.functional as F

# norm_first: for pre-norm formulation
# uses falsh-attenion viz fast and memory efficient 
encoder_block=nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

# Forward pass
out=encoder_block(src, src_mask=None, src_key_padding_mask=None, is_causal=False)
```

*TranformerDecoderLayer[](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html#transformerdecoderlayer)*

```python
import torch
import torch.nn as nn
import torch.functional as F

# norm_first: for pre-norm formulation
# uses falsh-attenion viz fast and memory efficient 
decoder_block=nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

# Forward pass
out=decoder_block(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False)
```