- 最初 batch_size 取 32 ，勉強能跑，但太慢了。~40min/epoch
  所以改到 batch_size=8。~30 sec/epoch

- VGG16使用crossEntropyLoss，Loss_val 只能收斂到~1.6

  改至resnet18 - crossEntropyLoss，Loss_val 能收斂至<0.2

- 我取的val set是由所有training set 中分離出20%抽取而成。

  神奇的是訓練到剛開始收斂時，Loss_val 會先低於 Loss_train。



## Test Result

![result1](./result2.png)

## t-SNE 