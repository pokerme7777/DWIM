
<h1 align="center">
  <strong>DWIM:</strong> Towards <em>Tool-aware Visual Reasoning</em><br>
  via <em>Discrepancy-aware Workflow Generation</em><br>
  &amp; <em>Instruct-Masking Tuning</em>
</h1>

<h1 align="center">
  🎉 Accepted by ICCV 2025 🎉
</h1>

<div align="center">
    <img src="assets/Method.svg" width="70%">
    <p></p>
</div>

<div align="center">
    <a href="https://github.com/pokerme7777/DWIM/issues">
        <img src="https://img.shields.io/github/issues/pokerme7777/DWIM?style=flat-square">
    </a>
    <a href="https://github.com/pokerme7777/DWIM/network/members">
        <img src="https://img.shields.io/github/forks/pokerme7777/DWIM?style=flat-square">
    </a>
    <a href="https://github.com/pokerme7777/DWIM/stargazers">
        <img src="https://img.shields.io/github/stars/pokerme7777/DWIM?style=flat-square">
    </a>
<!--     <a href="https://arxiv.org/abs/2502.00372">
        <img src="https://img.shields.io/badge/xxxx.svg?style=flat-square">
    </a> -->
</div>

**This repo is the official implementation for the paper [DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning](https://arxiv.org/abs/xxxxxx).**

The code will be in July 2025.

## 🎥 4 mins quick intro about DWIM  
[![DWIM Video](https://img.youtube.com/vi/TJhJTfpAG7g/0.jpg)](https://www.youtube.com/watch?v=TJhJTfpAG7g)


## Abstract

Visual reasoning (VR), which is crucial in many fields for enabling human-like visual understanding, remains highly challenging. Recently, compositional visual reasoning approaches, which leverage the reasoning abilities of large language models (LLMs) with integrated tools to solve problems, have shown promise as more effective strategies than end-to-end VR methods. However, these approaches face limitations, as frozen LLMs lack tool awareness in VR, leading to performance bottlenecks. While leveraging LLMs for reasoning is widely used in other domains, they are not directly applicable to VR due to limited training data, imperfect tools that introduce errors and reduce data collection efficiency in VR, and challenging in fine-tuning on noisy workflows. To address these challenges, we propose DWIM: i) Discrepancy-aware training Workflow generation, which assesses tool usage and extracts more viable workflows for training; and ii) Instruct-Masking fine-tuning, which guides the model to only clone effective actions, enabling the generation of more practical solutions. Our experiments demonstrate that DWIM achieves state-of-the-art performance across various VR tasks, exhibiting strong generalization on multiple widely-used datasets.

## References
If you find this work useful for your research, please consider citing it.
```bibtex
@article{ke2025dwim,
  title={DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation \& Instruct-Masking Tuning},
  author={Ke, Fucai and Leng, Xingjian and Cai, Zhixi and Khan, Zaid and Wang, Weiqing and Haghighi, Pari Delir and Rezatofighi, Hamid and Chandraker, Manmohan and others},
  journal={arXiv preprint arXiv:2503.19263},
  year={2025}
}
```
