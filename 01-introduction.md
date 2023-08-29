---
title: "Introduction"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you build machine learning pipelines for time-series analysis?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Introduce machine learning concepts applicable to time-series forecasting.
- Introduce Google's ```TensorFlow``` machine learning library for Python.
::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

This lesson is the third in a series of lessons demonstrating Python libraries
and methods for time-series analysis and forecasting.

The first lesson, 
[Time Series Analysis of Smart Meter Power Consmption Data](https://carpentries-incubator.github.io/python-pandas-power-consumption/), 
introduces datetime indexing features in the Python ```Pandas``` library. Other
topics in the lesson include grouping data, resampling by time frequency, and 
plotting rolling averages. 

The second lesson, 
[Modeling Power Consumption with Python](https://carpentries-incubator.github.io/python-modeling-power-consumption/),
introduces the component processes of the SARIMAX model for single-step 
forecasts based on a single variable.

This lesson builds upon the first two by applying machine learning processes
to build models with potentially greater predictive power against larger 
datasets. Relevant concepts include:

- Feature engineering
- Data windows
- Single step forecasts
- Multi-step forecasts

Throughout, the lesson uses Google's ```TensorFlow``` machine learning library
and the related Python API, ```keras```. As noted in each section of the lesson, the
code is based upon and is in many cases a direct implementation of the 
[Time series forecasting tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
available from the [TensorFlow](https://github.com/tensorflow/docs/blob/master/README.md) 
project. Per the documentation, materials available from the TensorFlow 
GitHub site are published using an 
[Apache 2.0](https://github.com/tensorflow/docs/blob/master/LICENSE)
license. 

> Google Inc. (2023) *TensorFlow Documentation.* Retrieved from [https://github.com/tensorflow/docs/blob/master/README.md](https://github.com/tensorflow/docs/blob/master/README.md).

This lesson uses the same dataset as the previous two. For more information
about the data, see the
[Setup](https://carpentries-incubator.github.io/python-modeling-power-consumption/)
page.

::::::::::::::::::::::::::::::::::::: keypoints 

- The ```TensorFlow``` machine learning library from Google provides many
algorithms and models for efficient pipelines to process and forecast
large time-series datasets.

::::::::::::::::::::::::::::::::::::::::::::::::

