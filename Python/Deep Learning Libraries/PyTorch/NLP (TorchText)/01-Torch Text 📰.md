## *torchtext.nn*

```python
import torchtext

# InProjectionContiner: Continaer to project query/key/value in MultiheadAttention
in_projection_container=torchtext.nn.InProjectionContainer(key,query,value)
k,q,v=in_projection_container(
	key=nn.Linear(n_embed,head_size)
	key=nn.Linear(n_embed,head_size)
	key=nn.Linear(n_embed,head_size)
)


# ScaledDotProduct() -> takes k,q,v as input from in_projection and perform scaled-dot-product returning attention-score
attention_layer=torchtext.ScaledDotProduct(dropout,batch_first=False)
attention_scores=scaledDotProduct(k,q,v,attn_mask=None)


# MultiHeadAttentionContainer
attentionBlock=torchtext.nn.MultiHeadAttenionContainer(
	nhead, in_proj_container, attention_layer, 
	out_proj=torch.nn.Linear(n_embed,n_embed), batch_first=False)

out=attentionBlock(k,q,v,attn_mask)
```

## *torchtext.utils*

```python
import torchtext

tokenizer=torchtext.utils.get_tokenizer(tokenizer,language='basic_english')
# tokenizer – the name of tokenizer function. If None, it returns split() function, which splits the string sentence by space. If basic_english, it returns _basic_english_normalize() function, which normalize the string first and split by space. If a callable function, it will return the function. If a tokenizer library (e.g. spacy, moses, toktok, revtok, subword), it returns the corresponding library.

# language – Default en


# Return an iterator that yields the given tokens and their ngrams.
itr=torchtext.utils.ngram_iterator(tokens_list,ngram)
```


## *torchtext.vocab*

```python
vocab_object=torchtext.vocab.vocab(ordered_dict, min_freq, specials Optional[List[str]]
, special_first: bool = True)

token_id=vocab[token]
token_id=vocab.lookup_indices(tokens)
token=vocab.lookup_tokens(token_ids)

# for oov token
vocab.set_default_index(vocab[unkn_tkn])


vocab=torchtext.vocab.build_vocab_from_iterator(iterator: Iterable, min_freq: int = 1, specials: Optional[List[str]] = None, special_first: bool = True, max_tokens: Optional[int] = None)
```

### Pretrained Word Embeddings[](https://pytorch.org/text/stable/vocab.html#pretrained-word-embeddings)

1. Glove[](https://pytorch.org/text/stable/vocab.html#glove)()
2. FastText[](https://pytorch.org/text/stable/vocab.html#fasttext)
3. CharNGram[](https://pytorch.org/text/stable/vocab.html#charngram)

## *torchtext.utils*

```python
import torchtext

# download data from url
torchtext.utils.download_from_url(url,root="./",overwrite=False)

# Extract zip files
torchtext.utils.extract_archive(from_path,to_path,overwrite=False)
```

## *torchtext.transforms[](https://pytorch.org/text/stable/transforms.html#module-torchtext.transforms)*

transforms are common text transforms. They can be chained together using [`torch.nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential "(in PyTorch v2.2)") or using [`torchtext.transforms.Sequential`](https://pytorch.org/text/stable/transforms.html#torchtext.transforms.Sequential "torchtext.transforms.Sequential") to support torch-scriptability.

## *torchtext.models*[](https://pytorch.org/text/stable/models.html#module-torchtext.models)

Pretrained models u can use for ur tasks

## `Step by step guide to train model`[]([SST-2 Binary text classification with XLM-RoBERTa model — Torchtext 0.18.0 documentation (pytorch.org)](https://pytorch.org/text/stable/tutorials/sst2_classification_non_distributed.html)) 