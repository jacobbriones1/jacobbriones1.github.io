---
title: "Can Attention Replace Convolution?"
date: 2021-01-27
tags: [attention, deep-learning]
permalink: /attention/
mathjax: true
---

# Introduction
Attention mechanisms have gained the interest of a large number of researchers. 
The [Transformer](https://arxiv.org/abs/1706.03762) introduced by Vaswani et al. in 2017 has been used extensively for text processing tasks due to it's ability to capture long-range dependencies. 
In recent works, it has been shown that attentional architectures can outperform many state of the art models for vision tasks such as object detection, classification, masking, etc. In fact, Cordonnier et al. proved that we can simulate **any convolutional neural network** with self-attention architectures. 

# Self-Attention and Convolution
<p align="center">
  *How does the Transformer operate in $2-D$*
 </p>

