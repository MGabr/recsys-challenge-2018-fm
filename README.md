# Factorization Machine  approach for the RecSys Challenge 2018

This is a Factorization Machine approach for the ACM RecSys Challenge 2018 based on [FM-spark](https://github.com/Intel-bigdata/FM-Spark).

It is however not making useful predictions since the number of latent factors is limited due to model size / memory 
constraints of the FM implementation. An adjusted FM implementation using a Parameter Server might solve these problems.
