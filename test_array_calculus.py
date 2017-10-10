#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import nose
import array_calculus as ac

def test_derivative():
	"""Testing our derivative matrix for three point of an exponential function"""
	# Testing for derivative of exponential at points x=0,1,2
	actual = np.array([(np.exp(1)-np.exp(0)), (np.exp(2)-np.exp(0))/2, (np.exp(2)-np.exp(1))])
	# Testing implementation
	def exponential():
		t = np.linspace(0,2,3)
		ex = np.vectorize(np.exp)
		ex = ex(t)
		return ex
	trial = np.dot(ac.derivative(0,2,3),exponential())
	# Debug message
	print("Should be: ",actual," but returned this: ",trial)
	for m in range(3):
		nose.tools.assert_almost_equal(actual[m],trial[m],4)

def test_second_derivative():
	"""Testing our second derivative matrix for five points of the exponential function"""
	# Testing the second derivative at x = 0,1,2,3,4
	actual = np.array([1.47625,6.96536,10.20501, 25.82899, 10.90807])
	# Testing impementation
	def exponential():
		t = np.linspace(0,4,5)
		ex = np.vectorize(np.exp)
		ex = ex(t)
		return ex
	trial = np.dot(ac.second_derivative(0,4,5),exponential())
	# Debug message
	print("Should be: ",actual," but returned this ",trial)
	for m in range(5):
		nose.tools.assert_almost_equal(actual[m],trial[m],4)
