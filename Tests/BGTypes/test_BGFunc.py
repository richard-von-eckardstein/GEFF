from src.BGQuantities.BGTypes import BGVal, BGFunc, BGSystem, Val, Func
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
        return 415.2975

    def sys(self):
        x = BGVal("x", 2, 0)
        y = BGVal("y", 1, 0)
        f1 = BGFunc("f1", [x], 2, 2)
        f2 = BGFunc("f2", [x, y], 3, 1)
        U = BGSystem([x, y, f1, f2], 0.5, 2.0)
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
    
    def test_class_single(self):
        x = BGVal("x", 1, 1)
        f = BGFunc("f", [x], 2, 2)
        assert f.name == "f"
        assert f.dtype == np.float64
        assert f.u_H0 == 2
        assert f.u_MP == 2
        for arg in f.Args:
            assert arg == x
    
    def test_init_single(self, func1, func2):
        f = self.inst(func1, 1)
        randval = random.random()
        assert f.GetBaseFunc()(randval) == func1(randval) 
        assert f.GetUnits() == True
        assert f.GetConversion() == 0.5**2 * 2.0**2
        assert len(f.GetArgConversions()) == 1
        for argconversion in f.GetArgConversions():
            assert argconversion == 0.5**2 * 2.0**0

        with pytest.raises(TypeError) as exc_info:
            f = self.inst(func2, 1)

        with pytest.raises(ValueError) as exc_info:
            f = self.inst(lambda x: "hi", 1)
            f = self.inst(lambda x: BGVal, 1)
            f = self.inst(lambda x: BGVal("x",2, 1), 1)

    #evaluate function
    def test_Call_single(self):
        pass

    






    #evaluate function
    #convert units

    #test mutli-variate function

    #test function returning multiple values (should raise Error)

    #test function returning a string or other bad datatype (should raise Error)
    pass