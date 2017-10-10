# PHYS220 CW 6

**Author(s):** **Jarod Penniman, Jared Love**

[![Build Status](https://travis-ci.org/chapman-phys220-2017f/cw-06-jareod.svg?branch=master)](https://travis-ci.org/chapman-phys220-2017f/cw-06-jareod)

## Specification

In this assignment, we will start going deeper into the tools we have already learned. Let us practice thinking in arrays more directly.

1. In a python module ```array_calculus.py``` create a central finite difference linear operator represented by a 2-dimensional matrix. That is, define a function ```derivative(a,b,n)``` that takes three input parameters, the two endpoints of the domain $[a,b]$ on which the derivative acts and the number of *points* $n$ into which that domain is partitioned (yielding $n-1$ intervals), and returns a single two-dimensional numpy array $D$ of dimension $n\times n$ that represents the finite difference operation. Done correctly, given a discrete function approximation represented by an $n$-dimensional array $f = [f_0, \ldots, f_{n-1}]$ defined as $f_i = f(x_i)$ on the grid of points $x_i$ from $i=0$ to $i=n$ with fixed spacing $x_{i+1} - x_i = dx$, where $dx = (b-a)/(n-1)$, applying the central difference matrix should yield a new $n$-dimensional array $Df = [(Df)_0, \ldots, (DF)_{n-1}]$ with interior points: $$(Df)_i = \frac{f_{i+1} - f_{i-1}}{2dx}.$$ Note that the numpy matrix product operation is ```@```, as opposed to ```*```, which performs component-wise products. For the boundary points, use the appropriate forward or backward finite difference formula instead of the central difference formula above. Write tests of your implementation in the module ```test_array_calculus.py``` to make sure it works as you expect. (Extra credit: Try to use features of numpy to avoid explicit for loops---see if you can define the derivative matrix entirely using clever manipulation of array features.)
1. In a Jupyter notebook, ```calculus.ipynb```, describe the problem, describe what the form of the derivative matrix should look like using $\LaTeX$ code, then demonstrate your solution by plotting the test functions: $f(x) = x^2$, $s(x) = \sin(x)$, and $g(x) = \exp(-x^2/2)/\sqrt{2\pi}$. For each function, make a separate plot of the function and its derivative together, with an appropriate legend labeling the curves. Title each plot in an informative way, and choose domains and ranges of each plot that sensibly show your results. (You should write your plotting code in the ```array_calculus.py``` module to be tidy, then import that module and run the appropriate functions in the notebook to generate the desired plots.) Explain whether the results match your intuition, and how changing the number of points $n$ affects your solutions.
1. Similar to the first problem, create a second-order central finite difference linear operator $D_2$ represented by a 2-dimensional matrix (i.e., repeat the first-order finite difference formula from above twice - what second-order formula do you get?). Write a function ```second_derivative(a,b,n)``` that returns a second-order derivative matrix  $D2$ using the explicit second-order different formulas. (Extra credit: Try to avoid explict for loops as in the first derivative implementation.) Is the returned matrix $D_2$ equal to the matrix product of $D$ from the previous exercise with itself? Write suitable tests in ```test_array_calculus.py``` to convince yourself that your code is working correctly. Update your plotting code to include the second-order derivative of each function in the same plots you made before. Update your notebook to discuss the second-order derivative, both in the equations, around the plots, and in your conclusions.
1. (Extra credit: Is it possible to define such a derivative operator for a non-uniform mesh spacing, meaning a domain that does not have equal spacings $dx$ between each point? If so, try it and plot your results!)

Pro-tip: using git to manage conflicts on Jupyter notebooks is a pain. I recommend delegating one person from your group to edit the notebook, to avoid merge conflicts.

## Assessment

Analyze in this section what you found useful about this assignment in your own words. Include any lingering questions or comments that you may have.

**This was a useful assignment in thinking about alternative ways to  define derivatives, or other linear operators, in computer science.**

## Honor Pledge

I pledge that all the work in this repository is my own with only the following exceptions:

* Content of starter files supplied by the instructor;
* Code borrowed from another source, documented with correct attribution in the code and summarized here.

Signed,

**Jarod Penniman, Jared Love**
