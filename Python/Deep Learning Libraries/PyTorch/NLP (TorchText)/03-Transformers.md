%%  %%![[Pasted image 20240325074323.png]]
## [Transformers in Deep Learning](https://arxiv.org/abs/1706.03762)

![[Pasted image 20240325075038.png]]

Transformers are a class of deep learning models that have revolutionized the field of natural language processing (NLP) and beyond. ==They are based on the encoder-decoder architecture, where the encoder converts an input sequence into a fixed-length vector, and the decoder generates an output sequence from the encoded representation.==

**Key Features:**

* **Attention Mechanism:** Transformers employ an attention mechanism that allows them to ==focus on specific parts of the input sequence when generating the output==. This enables them to capture long-range dependencies and context.
* **Self-Attention:** Transformers use self-attention to learn relationships within the input sequence, allowing them to model complex structures and patterns.
* **Parallel Processing:** Transformers can process the entire input sequence in parallel, making them highly efficient and suitable for large datasets.

**Types of Transformers:**

* **GPT (Generative Pre-trained Transformer):** A transformer model trained on a massive text dataset to generate human-like text and perform language-related tasks.
* **BERT (Bidirectional Encoder Representations from Transformers):** A transformer model that learns bidirectional representations of text, allowing it to understand context from both left and right.
* **T5 (Text-To-Text Transfer Transformer):** A transformer model that can handle a wide range of NLP tasks, including text summarization, question answering, and machine translation.

**Applications:**

Transformers have a wide range of applications in NLP, including:
* **Language Generation:** Generating text, dialogue, and code.
* **Machine Translation:** Translating text between different languages.
* **Question Answering:** Answering questions based on a given context.
* **Text Summarization:** Creating concise summaries of long text documents.
* **Named Entity Recognition:** Identifying and classifying entities (e.g., names, locations) in text.

Transformers have also been applied to other domains, such as:

* **Computer Vision:** Image classification, object detection, and segmentation.
* **Audio Processing:** Speech recognition, music generation, and sound classification.
* **Time Series Forecasting:** Predicting future values based on historical data.

**Advantages of Transformers:**

* **Long-range dependency modeling:** Captures relationships between distant elements in a sequence.
* **Parallel processing:** Processes entire sequences in parallel, improving efficiency.
* ==**Self-attention:**== Learns relationships within sequences, enabling complex modeling.
* **Transfer learning:** Can be fine-tuned for various NLP tasks with minimal data.
* **Generative capabilities:** Can generate human-like text and perform language-related tasks.
* **Multimodal capabilities:** Can be extended to handle tasks involving multiple modalities (e.g., text, images, audio).

**Future for transformers:**

![[Pasted image 20240325081016.png]]


## Self Attention:

**Problem with Word Embeddings:**

Word embeddings are vector representations of words that capture their semantic and syntactic properties. However, traditional word embeddings suffer from two main limitations:

* **Context-Insensitive:** They represent words with fixed vectors, which do not capture the different meanings a word can have depending on the context.
* **Positional Information Loss:** They do not encode the order or position of words in a sequence, which is crucial for understanding the meaning of a sentence.
* **Static in nature:** Word embeddings are generated during training and remain static, limiting their ability to fully grasp the context of a word within a specific test sentence.

**How Self-Attention Solves It:**

Utilizing self-attention enables the generation of intelligent contextual embeddings that possess both contextual awareness and dynamism, while incorporating positional encoding allows for the retention of positional information within the embeddings

Self-attention is a mechanism in transformers that addresses these limitations by:

* **Contextualized Representations:** Self-attention allows each word in a sequence to attend to and interact with other words in the same sequence. This enables the model to learn context-dependent representations of words.
* **Positional Encoding:** Transformers use positional encoding to inject positional information into the input sequence. This helps the model understand the order and relationships between words.

By combining self-attention and positional encoding, transformers can learn representations of words that are both contextually aware and positionally informative. This enables them to capture complex relationships and patterns in text, leading to improved performance on NLP tasks.

**How Self-Attention Works:**

- Self-attention = weighted embeddings to capture context

![[Pasted image 20240325092456.png]]

- self-attention help in generating contextualize representation that represent similarity of a word in a sequence with every other word. ===Dot-product tells similarity b/w 2 vector.===

![[Pasted image 20240325092555.png]]

- self-attention can be carried out in || which help in speed boosting.

![[Pasted image 20240325092901.png]]

![[Pasted image 20240325093016.png]]

- ==No Learning Parameters so far==: Means no learning + no dynamic nature. Without incorporating weights into our contextual embeddings, we lack learning capability and dynamism. To enhance adaptability to various data, it is essential to introduce weights that a neural network can learn. 

![[Pasted image 20240325094351.png]]

- We notice that every word within a sequence plays the roles of a query, key, and value. Nevertheless, employing a single vector for all three roles simultaneously at the same position is suboptimal, because it can limit the model's ability to differentiate between different aspects of the input sequence. To overcome this limitation, our approach involves utilizing separate instances of word embeddings to independently fulfil each of these roles.

![[Pasted image 20240325095149.png]]

- The 3 vectors query, key and value are formed from the word's embedding only but they are specialized to serve their specific task.

![[Pasted image 20240325100237.png]]

- Once we have created these query, key and value vector for each word (which will be formed from word's embedding) we will use them instead rather than using the embedding of the word at each place.

![[Pasted image 20240325100256.png]]

![[Pasted image 20240325100844.png]]

- To generate the query, key, and value vectors, we apply a linear transformation to the word's embedding. This transformation involves multiplying the embedding vector by a specific weight matrix. These weights act as adjustable parameters that enable the model to learn from the data and extract relevant information during the self-attention process.

![[Pasted image 20240325101220.png]]

- ==Note:== During the forward pass, the same weight matrix is used for each word to generate the query, key, and value vectors. In the backward pass, adjustments are made to the gradients of this weight matrix for each word, enabling the model to better fit the data.

![[Pasted image 20240325101531.png]]

 - ==Note:== These operations can still be performed in || and thus provide a speed boost. 



**Summary:**

Self-attention is a mechanism in transformers that allows each element in a sequence to attend to and interact with other elements in the same sequence. It operates as follows:

1. **Query, Key, and Value Matrices:** The input sequence is projected into three matrices: query (Q), key (K), and value (V).
2. **Dot-Product Attention:** The query matrix is multiplied by the transpose of the key matrix, resulting in an attention matrix. Each element in the attention matrix represents the similarity between a pair of elements in the sequence.
3. **Softmax:** A softmax function is applied to the attention matrix, which converts the similarity scores into probabilities. These probabilities indicate the importance of each element in the sequence for the current element.
4. **Weighted Sum:** The value matrix is multiplied by the attention probabilities, resulting in a weighted sum. This weighted sum is the output of the self-attention layer.

Self-attention allows the model to learn relationships between different parts of the sequence and focus on the most relevant elements for each position. This enables transformers to capture complex patterns and dependencies in the data.

**Example:**

Consider the sentence "The cat sat on the mat."

* **Input Sequence:** (The, cat, sat, on, the, mat)
* **Query Matrix (Q):** `[[Q1], [Q2], [Q3], [Q4], [Q5], [Q6]]`
* **Key Matrix (K):** `[[K1], [K2], [K3], [K4], [K5], [K6]]`
* **Value Matrix (V):** `[[V1], [V2], [V3], [V4], [V5], [V6]]`

The self-attention layer will calculate the attention matrix by multiplying Q and K^T. For example, the attention score between "cat" and "mat" would be calculated as Q2 * K6^T.

The softmax function would then convert these scores into probabilities, indicating how much attention "cat" pays to "mat" and vice versa.

Finally, the weighted sum would be calculated by multiplying V by the attention probabilities. This would result in a new representation of "cat" that takes into account its relationship with "mat" and other words in the sentence.

## Scaled Dot Product Attention:

**[Scaled Dot-Product Attention:](https://www.youtube.com/watch?v=r7mAt0iVqwo&list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&index=75&ab_channel=CampusX)**

In the context of transformer-based models and self-attention mechanisms, the dot product attention is often scaled to improve the stability and performance of the model. The scaling involves dividing the dot product of query and key vectors by the square root of the dimension of the key vectors.

**Formula:**

```
Attention (Q,K,C) without scale =softmax(Q*K^T)*V
Attention(Q,K,V) with scale=softmax((Q * K^T) / sqrt(dk)) * V

# Q: Query-matrix
# K: Key-matrix
# V: Value-matrix
# dk: dmn of K-matrix == dmn of Q-matrix == dmn of V-matrix

```

where:

* Q, K, V are the query, key, and value matrices, respectively
* dk is the dimension of the key vectors

**Why Do We Scale Self-Attention?**

Scaling the dot-product attention has two main benefits:

1. **Gradient Stability:** The dot-product operation can result in very large values, especially when the input vectors have high dimensionality. Scaling by the square root of the key dimension helps to stabilize the gradients and prevent the attention weights from becoming too large.
2. **Improved Performance:** Empirical evidence has shown that scaling the dot-product attention leads to better performance on NLP tasks. It helps to prevent the attention weights from being dominated by a few high-similarity pairs and allows the model to attend to a wider range of elements in the sequence.
3. **Stability:** Scaling helps prevent extremely large or small values in the dot products, which can lead to numerical instability during training.

4. **Balancing Signal Magnitudes:** By scaling, the dot products are normalized across different dimensions, ensuring that the magnitudes of attention weights are not too small or too large, which can affect the learning dynamics and convergence of the model.

5. **Attention Distribution:** Scaling influences the distribution of attention weights, making it more evenly spread and preventing any single dimension from dominating the attention computation.

**Example:**

Consider the following input vectors:

```python
Q = [0.5, 0.3, 0.2]
K = [0.4, 0.6, 0.8]
```

Without scaling, the dot-product attention would result in:

```python
Attention(Q, K, V) = softmax([0.2, 0.36, 0.56]) * V
```

However, with scaling by sqrt(dk) = sqrt(3) = 1.732, we get:

```python
Attention(Q, K, V) = softmax([0.116, 0.208, 0.324]) * V
```

As you can see, scaling reduces the attention weights and makes them more evenly distributed. This helps the model to attend to all three elements in the sequence more effectively.

## Geometric Intuition of Self-Attention in Transformers:

Self-attention in transformers can be interpreted as ==GRAVITY== as  a geometric operation in a high-dimensional space.

* Each element in the input sequence is represented by a vector in this space.
* Self-attention computes the similarity between these vectors using dot products.
* The resulting attention scores are normalized to form a projection matrix.
* This projection matrix maps the input vectors to a new subspace, where each vector represents a weighted sum of the original vectors, with the weights determined by the attention scores.

In this geometric view, self-attention:

* Projects the input vectors onto a subspace defined by their relationships.
* Creates contextual representations of each element, influenced by its connections to other elements.
* Captures long-range dependencies between elements, even if they are separated in the sequence.

## Why self-attention is called "self":

[Self-attention is called "self" because](https://www.youtube.com/watch?v=o4ZVA0TuDRg&list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&index=76&ab_channel=CampusX) it allows a model to attend to different parts of its own input sequence. This is in contrast to other attention mechanisms, such as encoder-decoder attention, where the model attends to a different sequence (e.g., a target sequence) given an input sequence.

With self-attention, the model can:

* Identify relationships between different elements within the input sequence.
* Create contextual representations of each element, taking into account its relationships with other elements.
* Capture long-range dependencies between elements, even if they are separated by other elements in the sequence.

The "self" in self-attention emphasizes that the model is attending to its own input, rather than to an external sequence. This allows transformers to model complex relationships within sequential data, which is crucial for tasks such as natural language processing, computer vision, and time series analysis.

Here is a simple analogy to illustrate self-attention:

Imagine you are reading a book and want to understand the meaning of a particular sentence. You might read the sentence multiple times, focusing on different words each time. This is similar to how self-attention works. The model reads the input sequence multiple times, attending to different elements each time, to create a deeper understanding of the sequence as a whole.