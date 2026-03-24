#### 介绍

自用kernel库

基于个人[CudaGemm](https://github.com/fw3569/CudaGemm)项目


#### 测试数据

transpose
```
kernel time : 1.9159ms, total time: 36.7215ms  
cublas kernel time: 1.91116ms, rate: 0.997528
```

reduce asum
```
kernel time : 6.85669ms, total time: 100.241ms
cublas kernel time: 7.6023ms, rate: 1.10874
```

elementwise geam
```
kernel time : 2.75101ms, total time: 51.1646ms
cublas kernel time: 2.78894ms, rate: 1.01379
```
