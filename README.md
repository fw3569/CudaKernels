#### 介绍

自用kernel库

基于个人[CudaGemm](https://github.com/fw3569/CudaGemm)项目

#### 测试数据

**transpose**

```
kernel time : 1.9159ms, total time: 36.7215ms  
cublas kernel time: 1.91116ms, rate: 0.997528
```

**reduce asum**

```
kernel time : 6.85669ms, total time: 100.241ms
cublas kernel time: 7.6023ms, rate: 1.10874
```

**elementwise geam**

```
kernel time : 2.75101ms, total time: 51.1646ms
cublas kernel time: 2.78894ms, rate: 1.01379
```

**softmax**

没什么有意义的测试数据  
cublas没有softmax，故使用了cudnn比较。但是cudnn的softmax比较特殊，使用<<<1,256>>>>的resident驻留kernel减少launch成本，小数组下更快，而大数组下性能相当差。这边也针对小数组使用了单kernel单block的策略，合并单kernel可以减少多余一次launch和读写global显存和重复计算exp的成本，单block可以进一步省略grid.sync或自旋锁的同步成本，但是计算耗时和register使用更多，在2^15下达到和cudnn的resident kernel相近的速度。  
在`2^25`下测试两个kernel的硬件throughout  
|kernel|compute throughout|memory throughout|
|--|--|--|
|expsum_kernel|11.46|97.97|
|divide_kernel|41.17|90.79|
