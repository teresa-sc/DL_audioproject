# Deep Learning Project: Finetuning, understanding and pruning of large audio models

This is meant as a starting point for your project. I will share some resources here and a notebook with a simple introduction example. 

You can always contact me at tksc@dtu.dk for any questions. We will meet every Monday 13-15 in building 321, room 232 for a joint supervision meeting, please prepare a small update of what you did and plan to do next week. 

For the first two weeks you should define the scope of your project (minimum: decide on a task and dataset) and start finetuning a model. Also, get familiar with the methods in the intro.ipnyb and maybe start a literature search of methods you want to apply later on (e.g. pruning, knowledge distillation, explainability). 

Your project should include: 
- finetuning of a pretrained large audio model (e.g. wavLM, wav2vec2, HuBERT, ...) to a downstream task (e.g. emotion recognition, speaker identification, automatic speech recognition, speech enhancement, speech separation, ...)
- analysis of hidden representations
- some form of pruning or knowledge distillation (KD)
- comparison of either several models, downstream tasks or pruning/KD approaches 
- optional: comparison to a baseline model (e.g. CNN) trained from scratch

---

The motivation for this project from my side are these two of my recent papers, you can use them as an inspiration and stick to them closely in your project but you are also free to come up with your own ideas!

Dorszewski, T., Tětková, L., & Hansen, L. K. (2024). Convexity-based Pruning of Speech Representation Models. arXiv preprint arXiv:2408.11858. https://arxiv.org/pdf/2408.11858 

Dorszewski, T., Jacobsen, A. K., Tětková, L., & Hansen, L. K. (2024). How Redundant Is the Transformer Stack in Speech Representation Models?. arXiv preprint arXiv:2409.16302. https://arxiv.org/pdf/2409.16302
