# Starting point: reading the Linear layer

## Context
2026-11-17. User invoked the teach skill and selected “Mainly understand existing code.”

## Stated prior knowledge and questions
The user has discussed MicroGPT's token vectors, normalization, and next-token cross-entropy. They now ask whether torch.empty zero-fills, what nn.Parameter means, how initialization std is chosen, and whether @ / .T are einops syntax.

## Lesson and insight targets
See `../lessons/0001-reading-linear.html`. Distinguish allocation, parameter registration, initialization, and forward computation; predict the result and shapes of a small matrix product.

## Evidence and next review
Questions identify the learning targets; mastery has not yet been demonstrated. Numerical example was verified with installed PyTorch. Ask for shape predictions and the meaning of empty at a later session before repeating explanations.
