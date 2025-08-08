from GEFF.BGTypes import BGVal, BGFunc, BGSystem, Val, Func
import pytest
import random
import numpy as np

class TestBGFunc():
    @pytest.fixture
    def func1(self):
        return lambda x: x**2

    @pytest.fixture
    def func2(self):
        return lambda x, y: x**2 + y**4
    
    @pytest.fixture
    def v1(self):
        return 41.2
    
    @pytest.fixture
    def v2(self):
        return 90.12

    def sys(self):
        x = BGVal("x", 1, 1)
        y = BGVal("y", 1, 0)
        f1 = BGFunc("f1", [x], 3, 1)
        f2 = BGFunc("f2", [x, y], 1, 3)
        U = BGSystem([x, y, f1, f2], 0.55, 0.32)
        return U
    
    def inst(self, func, nargs):
        U = self.sys()
        U.Initialise(f"f{nargs}")(func)
        return getattr(U, f"f{nargs}")
    
    def instarg(self, arg, val):
        U = self.sys()
        U.Initialise(arg)(val)
        return getattr(U, arg)
    
    #test single-variable function
    
    def test_class(self):
        #single-variable function
        x = BGVal("x", 2, 0)
        f = BGFunc("f", [x], 2, 2)
        assert f.name == "f"
        assert f.dtype == np.float64
        assert f.u_H0 == 2
        assert f.u_MP == 2
        for arg in f.Args:
            assert arg in [x]

        #multivariate function
        x = BGVal("x", 1, 1)
        y = BGVal("y", 1, 0)
        f = BGFunc("f", [x, y], 3, 1)
        assert f.name == "f"
        assert f.dtype == np.float64
        assert f.u_H0 == 3
        assert f.u_MP == 1
        for arg in f.Args:
            assert arg in [x, y]
    
    def test_init(self, func1, func2, v1, v2):
        #Initialise single-variable function
        f = self.inst(func1, 1)
        assert f.GetBaseFunc()(v1) == func1(v1) 
        assert f.GetUnits() == True
        assert f.GetConversion() == 0.55**3 * 0.32**1
        assert len(f.GetArgConversions()) == 1
        for argconversion in f.GetArgConversions():
            assert argconversion == 0.55**1 * 0.32**1

        #Initialise multivariate function
        f = self.inst(func2, 2)
        assert f.GetBaseFunc()(v1, v2) == func2(v1, v2) 
        assert f.GetUnits() == True
        assert f.GetConversion() == 0.55**1 * 0.32**3
        assert len(f.GetArgConversions()) == 2
        expectargconversion = [0.55**1 * 0.32**1, 0.55**1 * 0.32**0]
        for i, argconversion in enumerate(f.GetArgConversions()):
            assert argconversion == expectargconversion[i]

        #Check that instantiating func with wrong variable count raises error
        with pytest.raises(TypeError) as exc_info:
            f = self.inst(func2, 1)
        with pytest.raises(TypeError) as exc_info:
            f = self.inst(func1, 2)
        with pytest.raises(TypeError) as exc_info:
            f = self.inst(lambda x, y, z: 1.0, 1)
        with pytest.raises(TypeError) as exc_info:
            f = self.inst(lambda x, y, z: 1.0, 2)

        #Checkt that instantiating func which does not return an np.floating type raises error
        with pytest.raises(ValueError) as exc_info:
            f = self.inst(lambda x: "hi", 1)
        with pytest.raises(ValueError) as exc_info:
            f = self.inst(lambda x, y: "hi", 2)

    def test_Units(self, func1):
        f = self.inst(func1, 1)
        f.SetUnits(True)
        assert f.GetUnits() == True
        f.SetUnits(False)
        assert f.GetUnits() == False

    #evaluate function
    def test_Call_single(self, func1, v1):
        f = self.inst(func1, 1)
        x = self.instarg("x", v1)

        #With Units:
        f.SetUnits(True)
        assert f(v1) == func1(v1)
        x.SetUnits(True)
        assert f(x) == func1(v1)
        x.SetUnits(False)
        assert f(x) == func1(v1)

        #Without Units:
        f.SetUnits(False)
        assert f(v1) == func1(v1*f.GetArgConversions()[0])/f.GetConversion()
        x.SetUnits(True)
        assert f(x) == func1(v1)/f.GetConversion()
        x.SetUnits(False)
        assert f(x) == func1(v1)/f.GetConversion()

        #Check that error occurs when calling with BGVal and incorrect signature:
        y = self.instarg("y", v1)
        with pytest.raises(AssertionError) as exc_info:
            f.SetUnits(True)
            f(y)

        #Check call with multiple args:
        with pytest.raises(Exception) as exc_info:
            f(1.0, 1.0)

    def test_Call_multi(self, func2, v1, v2):
        f = self.inst(func2, 2)
        x = self.instarg("x", v1)
        y = self.instarg("y", v2)

        #With Units:
        f.SetUnits(True)
        assert f(v1, v2) == func2(v1, v2)
        #All permutations of units
        bools = [True, False]
        for boolx in bools:
            for booly in bools:
                x.SetUnits(boolx)
                y.SetUnits(booly)      
                assert f(v1, y) == func2(v1, v2)
                assert f(x, v2) == func2(v1, v2)
                assert f(x, y) == func2(v1, v2)

        #Without Units:
        f.SetUnits(False)
        #All permutations of units
        assert f(v1, v2) == func2(v1*f.GetArgConversions()[0],
                                   v2*f.GetArgConversions()[1])/f.GetConversion()
        #All permutations of units
        bools = [True, False]
        for boolx in bools:
            for booly in bools:
                x.SetUnits(boolx)
                y.SetUnits(booly)      
                assert f(v1, y) == func2(v1*f.GetArgConversions()[0],
                                   v2)/f.GetConversion()
                assert f(x, v2) == func2(v1,
                                   v2*f.GetArgConversions()[1])/f.GetConversion()
                assert f(x, y) == func2(v1, v2)/f.GetConversion()

        f.SetUnits(True)
        #Check that inverting order of args yields Error
        with pytest.raises(AssertionError) as exc_info:
            assert f(1/137, 2.0) == f(2.0, 1/137)

        #Check that error occurs when calling with BGVal and incorrect signature:
        with pytest.raises(AssertionError) as exc_info:
            f(y, x)

        #Check call with multiple args:
        with pytest.raises(Exception) as exc_info:
            f(1.0, 1.0, 2.0)
        
        #Check call with single args:
        with pytest.raises(Exception) as exc_info:
            f(1.0)