# Modern Self-Referential Weight Matrix

This is the official repository containing code for the paper:

[A Modern Self-Referential Weight Matrix That Learns to Modify Itself (ICML 2022 & NeurIPS 2021 Deep RL Workshop)](https://arxiv.org/abs/2202.05780)

An earlier/shorter version of the paper (only containing the RL part) was presented at [NeurIPS 2021 Deep RL Workshop](https://sites.google.com/view/deep-rl-workshop-neurips2021). The corresponding version is available on [Openreview](https://openreview.net/forum?id=lVUGfLpNpCF).

This reposity also contains code for the paper: [Accelerating Neural Self-Improvement via Bootstrapping (ICLR 2023 Workshop)](https://openreview.net/forum?id=SDwUYcyOCyP). Example scripts for this paper can be found under [supervised_learning/scripts/bootstrapping](https://github.com/IDSIA/modern-srwm/tree/main/supervised_learning/scripts/bootstrapping).

**Note(November 2023): We have a followup work on self-referential weight matrices (+ continual learning), [IDSIA/automated-cl](https://github.com/IDSIA/automated-cl/tree/main)**

## General instructions
Please refer to the readme file under each directory for further instructions.

License files can be found under the corresponding directories.

In all tasks, our custom CUDA kernels will be automatically compiled.
To avoid recompiling the code multiple times, we recommend to specify the path to a directory to store the compiled code via:
```
export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions/rl"
```

## BibTex
ICML 2022:
```
@inproceedings{irie2022modern,
  title={A Modern Self-Referential Weight Matrix That Learns to Modify Itself},
  author={Kazuki Irie and Imanol Schlag and R\'obert Csord\'as and J\"urgen Schmidhuber},
  booktitle={Proc. Int. Conf. on Machine Learning (ICML)},
  address={Baltimore, {MD}, {USA}},
  month=jul,
  year={2022}
}
```
NeurIPS 2021 Workshop:
```
@inproceedings{irie2021modern,
  title={A Modern Self-Referential Weight Matrix That Learns to Modify Itself}, 
  author={Kazuki Irie and Imanol Schlag and R\'obert Csord\'as and J\"urgen Schmidhuber},
  booktitle={Workshop on Deep Reinforcement Learning, NeurIPS},
  address={Virtual only},
  year={2021}
}
```
ICLR 2023 Workshop:
```
@inproceedings{irie2023accelerating,
  title={Accelerating Neural Self-Improvement via Bootstrapping},
  author={Kazuki Irie and J{\"u}rgen Schmidhuber},
      booktitle={Workshop on Mathematical and Empirical Understanding of Foundation Models, ICLR},
      address={Kigali, Rwanda},
      year={2023}
}
```

## Links
* Other recent works on fast weight programmers: 
    * [Linear Transformers are Secretly Fast Weight Programmers (ICML 2021)](https://arxiv.org/abs/2102.11174)
    * [Going Beyond Linear Transformers with Recurrent Fast Weight Programmers (NeurIPS 2021)](https://arxiv.org/abs/2106.06295)
    * [Neural Differential Equations for Learning to Program Neural Nets Through Continuous Learning Rules](https://arxiv.org/abs/2206.01649)
* [Jürgen Schmidhuber's AI blog post on Fast Weight Programmers (March 26, 2021)](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html).
