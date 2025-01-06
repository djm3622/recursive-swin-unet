# recursive-swin-unet

models:
- standard swin-unet | MSE loss
- MSE loss on final | MSE loss on each intermediate
- - swin-unet (xN)
- - swin-unet (hierarchy)
- - - (1-4-16, no skip to nextmodel)
- - - (1-4-16, standrd skip to nextmodel)
- - - (1-4-16, cross-attention skip to nextmodel)

*Test the on intermeidate idea early on to not waste too much training time on it.*