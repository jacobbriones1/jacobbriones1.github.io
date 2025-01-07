---
title: "Object Detection with Transformers"
date: 2021-02-03
tags: [machine learning, computer vision, transformers]
permalink: /obj_det_detr/
excerpt: "A complete guide to Facebook’s Detection Transformer (DETR) for Object Detection."
mathjax: true
---


# Object Detection with Transformers

*Published in The Startup on February 3, 2021*

A complete guide to Facebook’s Detection Transformer (DETR) for Object Detection.

![Example output of DETR](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*hGcCmvb0iyOrVOooytKB_w.png)

## Introduction

The DEtection TRansformer (DETR) is an object detection model developed by the Facebook Research team that cleverly utilizes the Transformer architecture. In this post, I’ll delve into the inner workings of DETR’s architecture to provide some intuition on its components. The accompanying Colab notebook for this tutorial can be found here:

[Object Detection Colab Notebook](https://colab.research.google.com)

Below, I’ll explain some of the architecture, but if you just want to see how to use the model, feel free to skip to the coding section.

## The Architecture

The DETR model consists of a pretrained CNN backbone (like ResNet), which produces a set of lower-dimensional features. These features are then formatted into a single set and added to a positional encoding, which is fed into a Transformer consisting of an Encoder and a Decoder, similar to the Encoder-Decoder transformer described in the original Transformer paper ([Attention Is All You Need](http://arxiv.org/abs/1706.03762)). The output of the decoder is then fed into a fixed number of Prediction Heads, each comprising a predefined number of feed-forward networks. Each output from these prediction heads includes a class prediction and a predicted bounding box. The loss is calculated by computing the bipartite matching loss.

![DETR Architecture from https://arxiv.org/pdf/2005.12872v3.pdf](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*mWv8_woRz5bB-jnY.png)

The model makes a predefined number of predictions, and each of the predictions is computed in parallel.

### The CNN Backbone

Assume that our input image \( x_{\text{im}} \) has a height \( H_0 \), width \( W_0 \), and three input channels. The CNN backbone consists of a (pretrained) CNN (usually ResNet), which we use to generate \( C \) lower-dimensional features with width \( W \) and height \( H \) (In practice, we set \( C=2048 \), \( W=W_0/32 \), and \( H=H_0/32 \)).

This process yields \( C \) two-dimensional features. Since we will be passing these features into a transformer, each feature must be reformatted to allow the encoder to process each feature as a sequence. This is done by flattening the feature matrices into an \( H \times W \) vector and then concatenating each one.

[Output of Convolutional Layer → Image Features](https://miro.medium.com/v2/resize:fit:828/format:webp/0*S5jw7rKuvReCeLxk.png)

The flattened convolutional features are added to a spatial positional encoding, which can either be learned or predefined.

### The Transformer

The transformer is nearly identical to the original encoder-decoder architecture. The difference is that each decoder layer decodes each of the \( N \) (the predefined number of) objects in parallel. The model also learns a set of \( N \) object queries, which are (similar to the encoder) learned positional encodings.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*cLjhFcQXKyq4akSO.png)

### Object Queries

The figure below depicts how \( N=20 \) learned object queries (referred to as prediction slots) focus on different areas of an image.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*cluaAVtmTKTqwrRuTNG9_w.png)

> “We observe that each slot learns to specialize on certain areas and box sizes with several operating modes.” — The DETR Authors

An intuitive way to understand the object queries is by imagining that each object query is a person. Each person can ask, via attention, about a certain region of the image. So, one object query will always ask about what is in the center of an image, another will always ask about what is on the bottom left, and so on.

## Simple DETR Implementation with PyTorch

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class SimpleDETR(nn.Module):
    """
    Minimal Example of the Detection Transformer model with learned positional embedding
    """
    def __init__(self, num_classes, hidden_dim, num_heads,
                 num_enc_layers, num_dec_layers):
        super(SimpleDETR, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        # CNN Backbone
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Transformer
        self.transformer = nn.Transformer(hidden_dim, num_heads,
                                          num_enc_layers, num_dec_layers)

        # Prediction Heads
        self.to_classes = nn.Linear(hidden_dim, num_classes+1)
        self.to_bbox = nn.Linear(hidden_dim, 4)

        # Positional Encodings
        self.object_query = nn.Parameter(torch.rand(100, hidden_dim))
```
Note: The code snippet above is a minimal example and may require additional components and adjustments for a fully functional implementation.

For a more comprehensive understanding and practical implementation, refer to the Object Detection Colab Notebook.

### Bipartite Matching Loss (Optional)
Let $\hat{y} =\\{\hat{y}_i| i=1,…N\\}$ be the set of predictions where $\hat{y}=(\hat{c}_i, b_i)$ is the tuple consisting of the predicted class (which can be the empty class) and a bounding box $b_i=(\bar{x}_i, \bar{y}_i, w_i, h_i)$ where the bar notation represents the midpoint between endpoints, and $w_i$ and $h_i$ are the width and height of the box, respectively.

Let y denote the ground truth set. Suppose that the loss between y and ŷ is L, and the loss between each yᵢ and ŷᵢ is Lᵢ. Since we are working on the level of sets, the loss L must be permutation invariant, meaning that we will get the same loss regardless of how we order the predictions. Thus, we want to find a permutation σ∈ Sₙ which maps the indices of the predictions to the indices of the ground truth targets. Mathematically, we are solving for



## Conclusion
DETR represents a significant advancement in object detection by integrating Transformers into the architecture, allowing for a more straightforward and efficient detection pipeline. By understanding its components and implementation, one can appreciate the innovation it brings to the field of computer vision.

Written by Jacob Briones

Follow Jacob Briones on Medium

Published in The Startup
