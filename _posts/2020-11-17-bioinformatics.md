---
title: "Why We Love Alcohol"
date: 2020-11-17
tags: [biology, data science, bioinformatics]
permalink: /welovebooze/
excerpt: "What DNA Sequences Reveal"
mathjax: true
---

<p align="center">
  <img src="https://sayingimages.com/wp-content/uploads/drinking-shut-up-liver-meme.jpg" height="350" width="400" alt>
</p>

# Introduction
2020 has been one of the most stressful years of my life, and I (like many others I assume) wouldn't have been able to make it through without alcohol.
Alcohol consumption has increased signifiantly as a result of the stay at home orders, and public distress. 
But why is this? Am I a horrible person for downing an entire bottle of cheap gas-station merlot after work? 
Perhaps.. But despite what most religious organizations would have you believe, the desire to drinking alcohol is encoded in our DNA.
In fact, our interest in alcohol dates back over 6,000 years. But how did alcohol become part of our biological history? 
The answer to this question can be understood by understanding *yeast*, one of the first organisms that humans domesticated. 
In particular, we will take a look at a certain species of yeast called *Saccharomyces cerevisiae*, which is capable of brewing wine.

# The Diauxic Shift
The yeast *S. cerevisiae* often live on grapevines. 
It converts **glucose** found in fruit into **ethanol**.
In hopes of understanding let us ask the following question:*Why do crushed grapes need to be stored in tightly sealed barrels in order to produce alcohol?*
More importantly, once the supply of glucose runs out, how does the *S. cerevisiae* survive? 
It survives by *inverting* it's metabolism and using the ethanol which it produced as a source of energy. 
Imagine if humans could do this. <br>

This metabolic inversion is known as *diauxic shift*. 
And the reason why crushed grapes need to be stored in tightly sealed containers is due to the fact that diauxic shift can only occur in the absence of oxygen.
Neat! Are other organisms capable of this kind of metabolic inversion? No idea. 
But one way to find out is to analyze the genes which are responsible for the diauxic shift in *S. cerevisiae*.

# Which Yeast Genes are Respobsible for Diauxic Shift?
Let's conduct a thought experiment. 
We have a list of yeast genes $$\{g_1,\dots,g_n\}$$, and we are going to record the *gene expression level* of each $$g_i$$ starting six hours before the diauxic shift, and repeating the measurement every two hours until six hours after the shift has occured. 
Let $$m=\{-6,-4,-2,0,2,4,6\}$$ denote these hour checkmarks. 
We can arrange these measurements in the form of a matrix $$E$$, known as the *gene expression matrix*, where the $$(i,j)$$-entry of $$E$$ is given by 
$$
\begin{equation*}
E_{i,j}=\text{gene expression level of }g_i \text{ at time j}
\end{equation*}
$$

