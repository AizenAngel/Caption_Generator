# Caption Generation in PyTorch.

Code completely taken from [here]("https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning?fbclid=IwAR1yrSMIODRjfs4Nj12lhexf1Djn5J3d23-GTFNkymDYUvaeAAkddlGWnAQ"),  with resolved issues that came with using deprecated/moved Python functions.  
Link to paper [Show, Attend and Tell]("https://arxiv.org/abs/1502.03044")

## Setup
- Download the [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip) and [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip) images inside ```Image folder```

- Download [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) inside ```caption data``` folder. This zip file contain the captions. You will also find splits and captions for the Flicker8k and Flicker30k datasets, so feel free to use these instead of MSCOCO if the latter is too large for your computer.

- Run ```Scripts/create_input_files.py``` script.

- Run ```Scripts/train.py``` script and be patient ;)