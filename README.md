# T-ITS-PFNC
--------------
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**P**arameter-**F**ree **N**on-**C**onvex low-rank tensor completion, PFNC

>This is the code repository for paper 'A Parameter-free Nonconvex Low-rank Tensor Completion Model for Spatiotemporal Traffic Data Recovery' which is submitted to IEEE Transactions on Intelligent Transportation Systems.

## Overview
This project provides code implementation about how to use the Parameter-Free Non-Convex Tensor Completion model (TC-PFNC) and its robust extension(RTC-PFNC) to achieve accurate and robust traffic data recovery. We defined a parameter-free nonconvex regularizer and utilized it to construct two low-rank tensor completion models, aiming to improve the **precision**, **applicability**, and **robustness** of traffic data recovery.

<!-- ## Model description
We define a log-based nonconvex regularizer to approximate tensor algebraic rank, which can also simultaneously increase the punishment on noise and decrease the punishment on structural information. Specially, the regularizer does not involve any parameter.
 -->
## Datasets
In this repository, we have used four real-world traffic datasets to show how to implement our model, they are:

- **[Guangzhou urban traffic speed data set](https://doi.org/10.5281/zenodo.1205228)**: Guangzhou urban traffic speed data set. This data set contains traffic speed collected from 214 road segments over two months (from August 1 to September 30, 2016) with a 10-minute resolution (i.e., 144 time intervals per day) in Guangzhou, China. The tensor size is 214 × 144 × 61.
- **[PeMs freeway traffic volume data set](https://github.com/VeritasYin/STGCN_IJCAI-18)**: This data set contains traffic volume collected from 228 loop detectors with a 5-minute resolution (i.e., 288 time intervals per day) over the weekdays of May and June, 2012 in District 7 of California by Caltrans Performance Measurement System (PeMS). The tensor size is 228 × 288 × 44.
- **[Seattle freeway traffic speed data set](https://github.com/zhiyongc/Seattle-Loop-Data)**: This data set contains freeway traffic speed from 323 loop detectors with a 5-minute resolution (i.e., 288 time intervals per day) over the first four weeks of January, 2015 in Seattle, USA. The tensor size is 323 × 288 × 28.
- **[Birmingham parking occupancy data set](https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham)**: This data set registers occupancy (i.e., number of parked vehicles) of 30 car parks in Birmingham City for every half an hour between 8:00 and 17:00 over more than two months (77 days from October 4, 2016 to December 19, 2016). The tensor size is 30 × 18 × 77.

The datasets is also avaliable in [Transdim](https://github.com/xinychen/transdim).

## Recovery Performance
- **TC-PFNC**

![example](https://github.com/YoungHe49/T-ITS-PFNC/blob/main/Figures/0s-G-NM-0.6-link103.png)
  *(a) Time series of actual and estimated speed within two weeks from August 1 to 14.*

![example](https://github.com/YoungHe49/T-ITS-PFNC/blob/main/Figures/0s-S-NM-0.6-link9.png)
  *(b) Time series of actual and estimated speed within two weeks from January 1 to 14.*
