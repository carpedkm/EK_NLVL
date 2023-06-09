EK-NLVL
=====
PyTorch implementation of EK-NLVL QAS model



[Daneul Kim](flytodk98@gm.gist.ac.kr), [Daechul Ahn](daechulahn@gm.gist.ac.kr), [Jonghyun Choi](jc@yonsei.ac.kr)

[Abstract]
In this paper, we propose the framework of External Knowledge Transfer in Natural Language Video Localization (EK-NLVL). By utilizing the pretrained image captioner and unsupervised event proposal module, we generate pseudo-sentences and event proposals to train the Natural Language Video Localization (NLVL) model. Most of the existing approaches rely on costly annotations on sentences and temporal event proposals, restricting the models' performance only on the given datasets, not applicable to real-world NLVL problems.
The proposed EK-NLVL leverages the idea of generating the pseudo-sentences from the given frames and summarizes it to ground the video event.
We also propose the data augmentation with visual-aligned sentence filtering technique for pseudo-sentence generation that could effectively provide additional signal to the model for NLVL.
Moreover, we propose the simpler model that leverages similarity between frame and pseudo-sentence by using CLIP loss, which effectively uses External Knowledge Transfer for the NLVL task.
Experiments on Charades-STA and ActivityNet-Caption datasets demonstrate the efficacy of our method compared to the existing models.

#### Update
- published in Journal of KIISE 2022
- updated with link to pre-trained captioning model

#### Dependencies
This repository is implemented based on [PyTorch](http://pytorch.org/) with Anaconda.</br> from [LGI](https://github.com/JonghwanMun/LGI4temporalgrounding)
Refer to [Setting environment with anaconda](anaconda_environment.md) or use **Docker** (carpedkm/ektnlvl:latest).

### Preparation
- [Pre-trained captioner](https://github.com/carpedkm/EKT-NLVL_vidcaps)
- [Dataset](https://github.com/JonghwanMun/LGI4temporalgrounding)

#### Evaluating pre-trained models
* Using **anaconda** environment
```bash
conda activate tg

# Evaluate QAS model trained from ActivityNet Captions Dataset
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval \
                     --config [ANET CONFIG PATH] \
                     --checkpoint [ANET CHECKPOINT PATH] \
                     --dataset anet \
                     --ann_path <annotation> \
                     --exp_info <exp information>
# Evaluate QAS model trained from Charades-STA Dataset
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval \
                     --config [CHARADES CONFIG PATH] \
                     --checkpoint [CHARADES CHECKPOINT PATH] \
                     --dataset charades \
                     --ann_path <annotation> \
                     --exp_info <exp information>
```


```
@inproceedings{kim2022qas,
    title     = "{Utilizing External Knowledge Transfer in Natural Language Video Localization}",
    author    = {Kim, Daneul and Ahn, Daechul and and Choi, Jonghyun},
    booktitle = {preprint},
    year      = {2022}
}
```
