# recursive-swin-unet

## TODO
- standard swin-unet | MSE loss
- MSE loss on final | MSE loss on each intermediate (weighted towards the last one)
- - swin-unet (xN)
- - - (no skip)
- - - (standard skip)
- - - (cross attention skip)
- - swin-unet (hierarchy)
- - - (1-4-16, no skip to nextmodel)
- - - (1-4-16, standrd skip to nextmodel)
- - - (1-4-16, cross-attention skip to nextmodel)

*Test the on intermeidate idea early on to not waste too much training time on it.*

Need to figure out a way to encourage *something* where uncertainty is in the model.

## IDEAS
- Output a prediction + uncertainty mask at each level
- Use an attention mechanism to help subsequent levels focus on high-uncertainty regions
- Consider using a learned weighting scheme rather than fixed weights 

Each UNet outputs prediction, uncertainty value, and uncertainity mask. *swin (encouraged to minmize MSE) -> uncertainty mask -> any space with uncertainty is passed forward* -> intermediate subspaces are back prop on loss -> continues until end.* In this case the is no need to weight, but maybe?

**uncertainty budget** (see what this is, good/not idk)
```
prediction_loss = MSE(pred, target) * (1 - uncertainty_mask)
uncertainty_penalty = alpha * mean(uncertainty_mask)  # controls the penalty of being uncertain
total_loss = prediction_loss + uncertainty_penalty
```