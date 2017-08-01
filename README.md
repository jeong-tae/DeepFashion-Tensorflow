# DeepFashion-Tensorflow

This repository contains implementation of DeepFashion[]. But tensorflow doesn't support ROI pooling/Landmark pooling which is most important part of this paper.

If updated version support this pooling layer, will be updated repository also.
before the supporting, This repo only compute category and attribute loss.
To compute triplet loss, we need relevant and non-relevant set but no exists here. Relevant/Non-relevant set arbitrarily extracted random samples from the same/other class.

Currently this repo only predict categories range in 1 to 50. There is a error with attribute loss computation. i will figure out ASAP. top1 acc is about 25%, top5 is about 58%. I will found some way to improve and to reproduce original one.
After applying triplet loss, still no improvement yet.
