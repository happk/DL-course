# Assignment #3: AutoEncoder

## 資料集簡介

由於本次作業要求使用`作業2`的影像作為重建目標。

故使用

- training set: `"D:\Desktop\中興\碩二上\深度學習\作業2\DRIVE\training\images"`
  - 565*584

- test set: `"D:\Desktop\中興\碩二上\深度學習\作業2\DIRVE_TestingSet"`
  中的`01_real_A.png` | 序號 01~19
  - 512*512




## 模型架構

　　做segmentation時我先將圖片預處理為灰階圖，再輸入進模型，所以只需要channel=1。

而autoencoder的輸出則是原圖，所以我直接將RGB的原圖輸入模型，此時channel=3。

代碼3-1中，我這裡直接將pool + enc4: 64 x 64 x 512 作為latent space，實際上這與AE要降維提取特徵的初衷相悖，不過作業要求"作業二所繳交的Unet為修改基礎"故不做過多更動。

代碼3-2中，重新設計模型結構，latent space: 32x32x1024。

> 由於我在作業2時的Unet的encoder錯誤地沒有將feature map逐步縮小，故在代碼3-2中修正。

### 作業2中的Unet

```python
class UNet(nn.Module):
  def __init__(self):
      super(UNet, self).__init__()
      
      # Encoder
      self.enc1 = self._double_conv(1, 64)
      self.enc2 = self._double_conv(64, 128)
      self.enc3 = self._double_conv(128, 256)
      self.enc4 = self._double_conv(256, 512)
      
      # Decoder
      self.dec4 = self._double_conv(512 + 256, 256)
      self.dec3 = self._double_conv(256 + 128, 128)
      self.dec2 = self._double_conv(128 + 64, 64)
      self.dec1 = nn.Conv2d(64, 1, kernel_size=1)
      
      self.pool = nn.MaxPool2d(2)
      self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      
  def _double_conv(self, in_channels, out_channels):
      return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
      )
  
  def forward(self, x):
      # Encoder
      e1 = self.enc1(x)
      e2 = self.enc2(self.pool(e1))
      e3 = self.enc3(self.pool(e2))
      e4 = self.enc4(self.pool(e3))
      
      # Decoder
      d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
      d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
      d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
      
      return torch.sigmoid(self.dec1(d2))
```

### 作業3(本次)的AE

#### 3-1

```python
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = self._double_conv(3, 64)    # 修改輸入通道為3
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Decoder
        self.dec4 = self._double_conv(512 + 256, 256)
        self.dec3 = self._double_conv(256 + 128, 128)
        self.dec2 = self._double_conv(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, 3, kernel_size=1)  # 修改輸出通道為3
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
        
        return torch.sigmoid(self.dec1(d2))
```

#### 3-2

```python
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = self._double_conv(3, 64)      # 512x512x3  -> 512x512x64
        self.pool1 = nn.MaxPool2d(2, 2)           # 512x512x64 -> 256x256x64
        
        self.enc2 = self._double_conv(64, 128)    # 256x256x64 -> 256x256x128
        self.pool2 = nn.MaxPool2d(2, 2)           # 256x256x128 -> 128x128x128
        
        self.enc3 = self._double_conv(128, 256)   # 128x128x128 -> 128x128x256
        self.pool3 = nn.MaxPool2d(2, 2)           # 128x128x256 -> 64x64x256
        
        self.enc4 = self._double_conv(256, 512)   # 64x64x256 -> 64x64x512
        self.pool4 = nn.MaxPool2d(2, 2)           # 64x64x512 -> 32x32x512
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)  # 32x32x512 -> 32x32x1024
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 32x32x1024 -> 64x64x512
        self.dec4 = self._double_conv(1024, 512)  # 64x64x1024 -> 64x64x512 (包含skip connection)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # 64x64x512 -> 128x128x256
        self.dec3 = self._double_conv(512, 256)   # 128x128x512 -> 128x128x256
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # 128x128x256 -> 256x256x128
        self.dec2 = self._double_conv(256, 128)   # 256x256x256 -> 256x256x128
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # 256x256x128 -> 512x512x64
        self.dec1 = self._double_conv(128, 64)    # 512x512x128 -> 512x512x64
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)  # 512x512x64 -> 512x512x3
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)           # 512x512x64
        e2 = self.enc2(self.pool1(e1))   # 256x256x128
        e3 = self.enc3(self.pool2(e2))   # 128x128x256
        e4 = self.enc4(self.pool3(e3))   # 64x64x512
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))   # 32x32x1024
        
        # Decoder path
        d4 = self.up4(b)           # 64x64x512
        d4 = torch.cat([d4, e4], dim=1)  # 64x64x1024
        d4 = self.dec4(d4)         # 64x64x512
        
        d3 = self.up3(d4)          # 128x128x256
        d3 = torch.cat([d3, e3], dim=1)  # 128x128x512
        d3 = self.dec3(d3)         # 128x128x256
        
        d2 = self.up2(d3)          # 256x256x128
        d2 = torch.cat([d2, e2], dim=1)  # 256x256x256
        d2 = self.dec2(d2)         # 256x256x128
        
        d1 = self.up1(d2)          # 512x512x64
        d1 = torch.cat([d1, e1], dim=1)  # 512x512x128
        d1 = self.dec1(d1)         # 512x512x64
        
        out = self.final_conv(d1)   # 512x512x3
        return torch.sigmoid(out)
```



## Train & Evaluation

- loss: MSELoss()
- optimizer: Adam
- learning rate: 0.001

### 3-1

![training_curves](training_curves.png)

| Image    | PSNR     |
| -------- | -------- |
| Image_01 | 35.39373 |
| Image_02 | 34.3197  |
| Image_03 | 33.96246 |
| Image_04 | 33.73119 |
| Image_05 | 32.10983 |
| Image_06 | 34.77101 |
| Image_07 | 33.45338 |
| Image_08 | 32.77271 |
| Image_09 | 34.42336 |
| Image_10 | 35.27078 |
| Image_11 | 33.74523 |
| Image_12 | 33.73696 |
| Image_13 | 35.48898 |
| Image_14 | 33.25846 |
| Image_15 | 33.55208 |
| Image_16 | 34.36223 |
| Image_17 | 34.2885  |
| Image_18 | 34.70089 |
| Image_19 | 35.98142 |
| Average  | 34.17489 |

![reconstruction_01](autoencoder_results/reconstruction_01.png)

![reconstruction_02](autoencoder_results/reconstruction_02.png)

![reconstruction_03](autoencoder_results/reconstruction_03.png)

![reconstruction_04](autoencoder_results/reconstruction_04.png)

![reconstruction_05](autoencoder_results/reconstruction_05.png)

![reconstruction_06](autoencoder_results/reconstruction_06.png)

![reconstruction_07](autoencoder_results/reconstruction_07.png)

![reconstruction_08](autoencoder_results/reconstruction_08.png)

![reconstruction_09](autoencoder_results/reconstruction_09.png)

![reconstruction_10](autoencoder_results/reconstruction_10.png)

![reconstruction_11](autoencoder_results/reconstruction_11.png)

![reconstruction_12](autoencoder_results/reconstruction_12.png)

![reconstruction_13](autoencoder_results/reconstruction_13.png)

![reconstruction_14](autoencoder_results/reconstruction_14.png)

![reconstruction_15](autoencoder_results/reconstruction_15.png)

![reconstruction_16](autoencoder_results/reconstruction_16.png)

![reconstruction_17](autoencoder_results/reconstruction_17.png)

![reconstruction_18](autoencoder_results/reconstruction_18.png)

![reconstruction_19](autoencoder_results/reconstruction_19.png)

### 3-2

> 有點奇怪，loss已經收斂了，但PSNR還在上升?

![3-2. loss curve](3-2.%20loss%20curve.png)

| Image    | PSNR     |
| -------- | -------- |
| Image_01 | 35.93597 |
| Image_02 | 35.37436 |
| Image_03 | 31.8849  |
| Image_04 | 35.51675 |
| Image_05 | 34.92626 |
| Image_06 | 35.38337 |
| Image_07 | 33.29728 |
| Image_08 | 35.3901  |
| Image_09 | 36.55123 |
| Image_10 | 35.92144 |
| Image_11 | 34.01836 |
| Image_12 | 34.87777 |
| Image_13 | 35.72583 |
| Image_14 | 35.72775 |
| Image_15 | 32.04605 |
| Image_16 | 36.26254 |
| Image_17 | 35.69571 |
| Image_18 | 36.18268 |
| Image_19 | 35.56333 |
| Average  | 35.06746 |

![reconstruction_01](3-2_autoencoder_results/reconstruction_01.png)

![reconstruction_02](3-2_autoencoder_results/reconstruction_02.png)

![reconstruction_03](3-2_autoencoder_results/reconstruction_03.png)

![reconstruction_04](3-2_autoencoder_results/reconstruction_04.png)

![reconstruction_05](3-2_autoencoder_results/reconstruction_05.png)

![reconstruction_06](3-2_autoencoder_results/reconstruction_06.png)

![reconstruction_07](3-2_autoencoder_results/reconstruction_07.png)

![reconstruction_08](3-2_autoencoder_results/reconstruction_08.png)

![reconstruction_09](3-2_autoencoder_results/reconstruction_09.png)

![reconstruction_10](3-2_autoencoder_results/reconstruction_10.png)

![reconstruction_11](3-2_autoencoder_results/reconstruction_11.png)

![reconstruction_12](3-2_autoencoder_results/reconstruction_12.png)

![reconstruction_13](3-2_autoencoder_results/reconstruction_13.png)

![reconstruction_14](3-2_autoencoder_results/reconstruction_14.png)

![reconstruction_15](3-2_autoencoder_results/reconstruction_15.png)

![reconstruction_16](3-2_autoencoder_results/reconstruction_16.png)

![reconstruction_17](3-2_autoencoder_results/reconstruction_17.png)

![reconstruction_18](3-2_autoencoder_results/reconstruction_18.png)

![reconstruction_19](3-2_autoencoder_results/reconstruction_19.png)

