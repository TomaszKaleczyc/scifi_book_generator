

<img src="data/cover.jpg" height=300></img><img src="data/cover2.jpg" height=300></img><img src="data/cover3.jpg" height=300></img>
# SciFi book generator

The purpose of this project is to build a decoder only transformer architecture from scratch and train it to autogenerate SciFi stories based on transcripts from pulp magazines. The approach taken in the code is inspired by the original [Attention is all you need](https://arxiv.org/abs/1706.03762) as well as [Andrej Karpathy's coding implementation](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Project structure 

```
├── data                        # project source data
├── environment                 # project dependencies 
└── src                         # project source code
    ├── dataset                 # dataset creation tools
    ├── model                   # model definitions
    └── tokeniser               # text tokenisers
```


## Resources

* Working environment pre-requisites: Ubuntu18.04 LTS / Python 3.6.9 / unzip / virtualenv / CUDA version >=11.6
* Dataset: [Kaggle SciFi Stories Text Corpus](https://www.kaggle.com/datasets/jannesklaas/scifi-stories-text-corpus?resource=download) - use: `make download-dataset` to collect
