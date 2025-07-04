![[VAE_tutorial_by_ZhangXin.pdf]]

vae的神经网络分两个部分,一个编码器,一个解码器,编码器用来预测隐变量的分布,解码器用来使预测的图像和原图像在像素级别相近?


vae这两个损失存在一些问题,首先重构损失用MSE,对于机器来说几乎一样的噪声但是在人看来却有明显的不同,所以我们的重构项不用L2损失而是直接使用GAN的损失,直接用判别器来判断?

另外一个KL散度损失,我们想要p(z|x)接近标准正态分布N(0,I),但是又不能完全是N(0,I),因为如果完全是的话那我们就没必要把图像编码到隐空间,直接使用标准正态分布解码就行了,这样生成出来的图像肯定是不符合预期的,所以我们将其与diffusion模型结合,将编码到隐空间的分布z一步一步加噪声最终变成标准正态分布,然后再让解码器去解码预测?还是说我们需要通过diffusion重构噪声再解码?

- 用 encoder 得到 `z`（不是标准正态）
- 然后把 `z` 当作 **x₀**，用 forward diffusion 加噪变成 `z_T ~ N(0,I)`
- 再让模型从 `z_T` 开始，逐步预测回 `z₀`，作为 decoder 输入


|模型结构|Diffusion 的作用|
|---|---|
|**VAE-Diffusion（如 VDM）**|Diffusion 替代 KL 项，建模 latent 空间分布|
|**Diffusion Decoder（如 eDiff-I）**|Diffusion 替代 VAE 的解码器，提升图像质量，重构用 diffusion 去噪生成图像|

|方式|思路|是否还有 decoder|
|---|---|---|
|VAE + GAN|用 GAN loss 替代 MSE|✅ 通常还有解码器|
|VAE + Diffusion（Decoder）|用 diffusion 替代解码器|❌ 不再有传统 decoder|
|VAE + Diffusion（Latent）|Diffusion 模拟 latent space 的正态分布演化|✅ 有 decoder，也可能是 DDPM decoder|


|组合类型|Diffusion 的作用|对应的损失项|
|---|---|---|
|**VAE + GAN**|保留 VAE 结构，用 GAN 判别器提升图像质量|GAN loss + KL|
|**VAE + Diffusion (latent modeling)**|用 diffusion 建模潜变量 `z` 的演化|重建 loss + diffusion latent loss|
|**VAE + Diffusion (decoder)**|编码器输出 `z`，由 diffusion 解码器生成图像|仅 diffusion loss，无 GAN / MSE|
