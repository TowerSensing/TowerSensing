# TowerSensing: Linearly Compressing Sketches with Flexibility

This repository contains all related code of our paper "TowerSensing: Linearly Compressing Sketches with Flexibility". 

## Introduction

A Sketch is an excellent probabilistic data structure, which records the approximate statistics of data streams by maintaining a summary. Linear additivity is an important property of sketches. This paper studies how to keep the linear property after sketch compression. Most existing compression methods do not keep the linear property.We propose TowerSensing, an accurate, efficient, and flexible framework to linearly compress sketches. In TowerSensing, we first separate a sketch into two partial sketches according to counter values. For the sketch with small counters,we propose a key technique called TowerEncoding to compress it into a hierarchical structure. For the sketch with large counters, we propose a key technique called SketchSensing to compress it using compressive sensing. We theoretically analyze the accuracy of TowerSensing. We use TowerSensing to compress 7 sketches and conduct two end-to-end experiments: distributed measurement and distributed machine learning. Experimental results show that TowerSensing outperforms prior art on both accuracy and efficiency, which achieves up to 100× smaller error and 5.1× higher speed than state-of-the-art Cluster-Reduce. All related codes are open-sourced anonymously. 

## About this repository

* `CPU` contains codes of TowerSensing and the related algorithms implemented on CPU platforms. 

* `DML` contains codes of TowerSensing and the related algorithms in a simulated distributed machine learning (DML) system. 

* `FPGA` contains codes of TowerSensing implemented on FPGA platforms.

* More details can be found in the folders.