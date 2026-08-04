# Style Demo

This page showcases all the typography capabilities supported by the blog: textbook-style multi-level headings, mathematical formulas, chemical equations, code, tables, and more. When writing a blog post, you can refer directly to this page's Markdown source.

## Multi-level headings

Heading levels are distinguished by font size and style; English fragments within headings are automatically rendered in italics.

### This is a level-three heading

Body content. When writing, you only need to write `#` headings normally — the layout is handled automatically by the design system.

#### This is a level-four heading

Level-four headings are used for subsections within a section, visually de-emphasized to widen the hierarchy gap.

## Mathematical formulas

Inline formula: the mass–energy equivalence $E = mc^2$, and Euler's identity $e^{i\pi} + 1 = 0$.

Block formula — the probability density function of the normal distribution:

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Multi-line alignment — the chain rule for backpropagation:

$$
\begin{aligned}
\frac{\partial L}{\partial W^{[l]}} &= \delta^{[l]} (a^{[l-1]})^T \\
\delta^{[l]} &= (W^{[l+1]})^T \delta^{[l+1]} \odot \sigma'(z^{[l]})
\end{aligned}
$$

Matrices and summation — Softmax and cross-entropy loss:

$$
\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \qquad
L = -\sum_{i=1}^{K} y_i \log \hat{y}_i
$$

## Chemical equations

Written with the mhchem package using `\ce{}`:

$$
\ce{2H2 + O2 ->[点燃] 2H2O}
$$

Reversible reactions and precipitate/gas symbols:

$$
\ce{CO2 + C <=> 2CO}, \qquad
\ce{CaCO3 + 2HCl -> CaCl2 + H2O + CO2 ^}
$$

Coordination compounds and biological metabolism:

$$
\ce{K4[Fe(CN)6]}, \qquad
\ce{ATP + H2O -> ADP + Pi + 能量}
$$

## Code

```python
import torch

def lora_forward(x, W0, A, B, alpha, r):
    """LoRA: W = W0 + (alpha / r) * B @ A"""
    return x @ W0.T + (alpha / r) * (x @ A.T @ B.T)
```

## Tables and checkboxes

| Method | Memory footprint | Trainable parameter ratio |
| --- | --- | --- |
| Full fine-tuning | High | 100% |
| LoRA | Medium | ~0.1–1% |
| QLoRA | Low | ~0.1–1% |

- [x] Selected topic, completed
- [ ] Selected topic, to be written

## Blockquotes and callouts

> Entities must not be multiplied beyond necessity. — William of Ockham

> After writing: create `xxx.md` in this directory, then change the corresponding entry to `- [x] [Title](./xxx)`.

## Epigraphs and marginal notes

Epigraph at the top of the page:

<div class="epigraph">
<p>I do not frame hypotheses, for hypotheses are the great enemy of scientific research.</p>
<footer>— Isaac Newton</footer>
</div>

Marginal notes in body paragraphs<span class="marginnote">This is a marginnote, placed in the right-hand margin on wide screens and automatically pulled back into the body text on narrow screens.</span>, suitable for supplementary explanations, sources, or asides.

Numbered, footnote-style sidenotes<span class="sidenote-number"></span><span class="sidenote">This is a sidenote, numbered automatically, like a footnote in the margin of a book.</span>, consecutive ones count up in sequence<span class="sidenote-number"></span><span class="sidenote">The second sidenote.</span>.
