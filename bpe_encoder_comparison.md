# BPE Encoder Optimization: My Design vs. Production Implementations

## 1. Background

After exploring optimization strategies for my BPE trainer, I moved on to optimizing the BPE encoder.

My earlier trainer design used four cooperating data structures:

- **Max-heap:** Select the globally most frequent adjacent pair.
- **Pair-frequency hashmap:** Maintain dynamically changing pair frequencies.
- **Pair-occurrence index:** Locate occurrences of selected pairs.
- **Array-based doubly linked list:** Support efficient local merges without shifting token positions.

The encoder presents a related but simpler problem.

During BPE training, pair frequencies change dynamically, requiring repeated global pair selection. During encoding, the merge rules and their priorities are already fixed.

Therefore, my goal is to avoid repeatedly traversing the entire learned merge list while preserving the exact merge ordering.

## 2. My Proposed BPE Encoder Architecture

My encoder design uses three primary data structures:

| Data structure | Purpose |
|---|---|
| Merge-priority hashmap | Map each learned merge pair to its priority |
| Min-heap | Select the highest-priority currently applicable pair occurrence |
| Array-based doubly linked list | Represent the evolving token sequence with stable indices |

### 2.1 Merge-Priority Hashmap

The learned merge list contains \(M\) rules, ordered by their creation during training.

Instead of repeatedly scanning the entire list, I want to support average-case constant-time lookup of a pair's learned priority.

This separates static merge-rule information from the dynamically changing state of each pre-token.

### 2.2 Min-Heap for Merge Selection

My initial approach involved traversing the learned merge list and looking up occurrences in a reverse index.

However, that still required examining all \(M\) rules for every pre-token.

I subsequently realized that a pre-token initially containing \(L\) bytes has only \(L-1\) adjacent-pair occurrences.

Rather than examining the entire learned vocabulary, I can focus on the currently applicable pairs.

My proposed min-heap orders candidate occurrences by their learned merge priority, with positional ordering used to resolve overlapping occurrences consistently.

This allows me to reason about merge selection without repeatedly scanning all learned rules.

### 2.3 Array-Based Doubly Linked List

I retain my earlier idea of representing a linked list through stable integer indices rather than individually allocated Python objects.

Conceptually, I maintain:

- Token values.
- Previous-node indices.
- Next-node indices.

When two adjacent tokens merge, I update their logical relationships without shifting the remaining sequence.

This provides constant-time predecessor and successor access and avoids the memory overhead of independently allocated linked-list nodes.

### 2.4 Lazy Invalidation

Merging a pair changes neighboring relationships, potentially invalidating candidate occurrences already stored in the heap.

Rather than immediately removing every invalid candidate from the heap, I allow outdated records to remain until they are examined.

I then validate whether the referenced nodes are still active, adjacent, and represent the expected pair.

This avoids requiring arbitrary heap deletions.

The main correctness requirement is that stale records must never result in an invalid merge.

## 3. Complexity Analysis

Let:

- \(L\): Initial byte length of a pre-token.
- \(M\): Total number of learned merge rules.
- \(m\): Number of successful merges.

### 3.1 Naïve Encoding

If I examine every learned rule and scan the current pre-token to locate applicable occurrences, the simplified complexity is:

\[
\boxed{O(LM)}
\]

This becomes expensive when \(M\) is large.

### 3.2 My Optimized Encoder

My proposed approach avoids the repeated \(O(M)\) traversal.

An initial pre-token has at most \(L-1\) adjacent-pair occurrences.

Each successful merge reduces the active token count by one:

\[
m\leq L-1
\]

A merge changes only a constant number of neighboring relationships.

Therefore, under the assumptions of constant-time local updates and expected constant-time hashmap lookups, the total number of candidate records can be bounded by \(O(L)\).

Heap operations cost \(O(\log L)\).

This gives an expected per-pre-token complexity of:

\[
\boxed{O(L\log L)}
\]

The bound excludes one-time preprocessing of the learned merge rules.

| Approach | Expected time complexity |
|---|---|
| Naïve full merge-list scanning | \(O(LM)\) |
| My heap-based encoder | \(O(L\log L)\) |

The main improvement is eliminating the direct dependence on \(M\) during individual pre-token encoding.

## 4. Comparison with Google SentencePiece

SentencePiece is particularly interesting because its production BPE encoder uses several architectural ideas similar to mine.

Its C++ implementation includes priority-based candidate selection, symbol relationships, and handling of outdated merge candidates.

Reference: `google/sentencepiece`, `src/bpe_model.cc`

| Component | My encoder | SentencePiece BPE encoder |
|---|---|---|
| Merge eligibility | Learned merge-priority hashmap | Model vocabulary and symbol lookup |
| Merge selection | Min-heap | Priority queue |
| Sequence representation | Stable-index arrays | Indexed symbol records |
| Neighbor navigation | `prev` / `next` arrays | Previous/next symbol references |
| Candidate invalidation | Lazy validation | Stale candidate validation |
| Merge updates | Local adjacency updates | Local symbol and adjacency updates |
| Implementation language | Python | C++ |

### 4.1 Similarities

My design and SentencePiece share several important principles.

**First, both avoid repeatedly scanning the complete learned merge list.**

Instead, merge candidates are evaluated using information associated with the current input.

**Second, both use priority-based candidate selection.**

This preserves the importance of the learned merge ordering.

**Third, both maintain relationships between neighboring symbols.**

A merge changes only a local portion of the active token sequence.

**Fourth, both must account for outdated merge candidates.**

The set of eligible adjacent pairs changes as merging proceeds.

These similarities suggest that my approach follows several established principles of efficient BPE encoding.

### 4.2 Differences

The most significant differences concern implementation details.

SentencePiece is implemented in C++, which substantially reduces the overhead associated with individual operations and memory allocation.

My current design is intended for Python, where heap entries, hash tables, and auxiliary objects may introduce significant overhead.

Additionally, SentencePiece's BPE implementation uses model-dependent symbols and scores, whereas my encoder must follow the byte-level merge semantics specified in my assignment.

Therefore, similar data structures do not imply identical tokenization behavior or runtime performance.

## 5. Comparison with Hugging Face Tokenizers

Hugging Face Tokenizers provides another useful production reference.

Its BPE implementation is written primarily in Rust and emphasizes efficient native execution.

Reference: `huggingface/tokenizers`

| Dimension | My encoder | Hugging Face Tokenizers |
|---|---|---|
| Primary implementation language | Python | Rust |
| Merge information | Explicit learned merge priorities | Model-defined BPE merge information |
| Sequence processing | My array-based linked-list design | Native token-sequence processing |
| Repeated inputs | No caching assumed in my complexity model | Supports BPE caching |
| Memory management | Python-managed objects and arrays | Native data structures |

One important distinction is caching.

My \(O(L\log L)\) analysis describes encoding a pre-token independently.

However, real corpora contain many repeated pre-tokens.

A production encoder may benefit from reusing previously computed results, reducing repeated work.

This introduces a separate performance dimension: optimizing the cost of an individual encoding versus optimizing throughput over an entire corpus.

## 6. Memory Complexity

My encoder requires auxiliary storage for:

- The current token sequence.
- Previous/next relationships.
- Pending heap candidates.

Under the assumptions of my candidate-count analysis, these require:

\[
\boxed{O(L)}
\]

auxiliary memory per pre-token.

The learned merge-priority hashmap is shared across pre-tokens and requires additional tokenizer-wide memory:

\[
O(M)
\]

Therefore, the combined auxiliary memory requirement is approximately:

\[
\boxed{O(M+L)}
\]

for one actively encoded pre-token, excluding the vocabulary's underlying byte strings and other implementation overhead.

This differs from my earlier BPE trainer, which required global mutable state covering many distinct pre-tokens.

## 7. Practical Performance Considerations

Although my design improves asymptotic complexity, several factors could affect real-world performance.

### 7.1 Short Pre-Tokens

Many pre-tokens contain only a few bytes.

For short inputs, initializing a heap and maintaining linked-list metadata may cost more than performing a simpler sequence-based operation.

Consequently, better asymptotic complexity does not guarantee lower latency.

### 7.2 Python Interpreter Overhead

Each hashmap lookup, heap operation, and object allocation introduces runtime overhead.

Even with an \(O(L\log L)\) algorithm, a Python implementation may be slower than a native implementation using a less favorable theoretical algorithm.

### 7.3 Memory Locality

My stable-index representation avoids individually allocated linked-list nodes and may improve cache locality.

However, Python containers and dynamically allocated heap records can still introduce substantial memory overhead.

### 7.4 End-to-End Throughput

The complete tokenizer also performs preprocessing, UTF-8 encoding, and output conversion.

Optimizing BPE merging only accelerates one portion of this pipeline.

Therefore, I should distinguish merge-processing latency from end-to-end tokenizer throughput when evaluating performance.

## 8. Final Comparison

| Criterion | My proposed encoder | Production implementations |
|---|---|---|
| Avoids full merge-list scans | Yes | Yes |
| Priority-based merge selection | Yes | Yes |
| Efficient local sequence mutation | Yes | Yes |
| Stable symbol references | Yes | Commonly used |
| Lazy invalidation | Yes | Used in SentencePiece |
| Expected per-pre-token complexity | \(O(L\log L)\) under stated assumptions | Implementation-dependent |
| Native execution | No | Yes |
| Repeated-input caching | Not included in current analysis | Supported by some implementations |
| Correctness | Requires testing | Established by production implementations |
| Real-world efficiency | Requires benchmarking | Optimized for production workloads |

## 9. Final Takeaways

My BPE encoder design has evolved from naïve rule-list scanning toward an architecture based on **priority-based candidate selection, stable symbol references, local adjacency updates, and lazy invalidation**.

Its expected \(O(L\log L)\) complexity removes the repeated \(O(M)\) merge-list traversal that originally concerned me.

Comparing my design with SentencePiece reveals substantial similarities in algorithmic principles, especially priority-based selection and local sequence modification.

However, theoretical complexity is only one component of performance.

The primary remaining engineering considerations are:

1. Verifying exact merge-ordering semantics, including overlapping occurrences.
2. Ensuring stale heap entries cannot cause incorrect merges.
3. Measuring Python data-structure overhead for short pre-tokens.
4. Separating per-pre-token performance from end-to-end corpus throughput.
5. Evaluating memory usage and cache locality.

**Overall conclusion:** My proposed BPE encoder is architecturally close to established production designs. The main challenge is no longer eliminating full merge-list scans, but demonstrating that my implementation preserves correctness while balancing heap overhead, memory efficiency, and practical execution speed.