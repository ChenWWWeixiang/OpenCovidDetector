# Read Me

## Table of Contents  
+ [Preamble](#Preamble)
+ [Sampling](#Sampling)
+ [How_it_Works?](#How_it_Works?)
+ [Expectation](#Expectation)

<a name="Preamble"/>

## Preamble
This program approximates the dimension (Hausdorff dimension) of a fractal using the [box counting method](https://en.wikipedia.org/wiki/List_of_fractals_by_Hausdorff_dimension).
The inspiration of this project came from [3blue1brown's video](https://www.youtube.com/watch?v=gB9n2gHsHN4).

<a name="Sampling"/>

## Sampling
You can try out the program with the provided sample images by calling this command: `python3 fractals.py sierpinsky.png`. 

<a name="How it Works?"/>

## How it Works?
This program is set to do a brute-force test for the fractal dimension of an object. It generates the factors of a nxn image, box-counts each iteration, and calculates the dimension between each iteration. Finding the true answer is left as an exercise to the user.

This program works best with large resolution fractals / fractals with many iterations. It can also be modified to work with simple 2-D binary arrays.

<a name="Expectation"/>

## Expectation
This is a winter break project. Please do not expect perfection from this project.
