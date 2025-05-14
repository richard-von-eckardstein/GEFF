from src.BGQuantities.BGTypes import BGVal, BGFunc, BGSystem, Val, Func
import pytest
import numpy as np

class TestBGVal():
    @pytest.fixture
    def v1(self):
        return 10.
    
    @pytest.fixture
    def v2(self):
        return 113.2
    
    @pytest.fixture
    def a1(self):
        return np.array([10., 381., 2.329])
    
    @pytest.fixture
    def a2(self):
        return np.array([3., 89.2, 15.123])

    def sys(self):
        x = BGVal("x", 2, 1)
        U = BGSystem([x], 0.5, 2.0)
        return U
    
    def inst(self, val):
        U = self.sys()
        U.Initialise("x")(val)
        return U.x
    
    def test_Class(self):
        x = BGVal("x", 2, 1)
        assert x.name == "x"
        assert x.dtype == np.float64
        assert x.u_H0 == 2
        assert x.u_MP == 1

    def test_Init(self, v1):
        x = self.inst(v1)
        assert x.massdim==3
        assert np.issubdtype(x.value.dtype, np.floating)
        assert x.GetUnits() == True
        assert x.GetConversion() == 0.5**2 * 2.0

    def test_InitFroma1(self, a1):
        x = self.inst(a1)
        assert (x.value == a1).all()

    def test_Units(self, v1):
        x = self.inst(v1)
        x.SetUnits(True)
        assert x.GetUnits()
        assert x.value == v1

        x.SetUnits(False)
        assert not(x.GetUnits())
        assert x.value == v1/(0.5**2 * 2.0)

    def test_str(self, v1):
        x = self.inst(v1)
        x.SetUnits(True)
        assert x.__str__() == f"{x.name} (Unitful): {v1}"

        x.SetUnits(False)
        assert x.__str__() == f"{x.name} (Unitless): {v1/(x.GetConversion())}"

    def test_getitem(self, a1):
        x = self.inst(a1)
        assert (a1 == np.array([x[i] for i in range(len(a1))])).all()

    def test_len(self, v1, a1):
        x = self.inst(a1)
        assert len(x) == len(a1)

        #len of float is not defined
        x = self.inst(v1)
        with pytest.raises(Exception) as exc_info:
            len(x)

    def test_abs(self, v1):
        x = self.inst(v1)
        assert abs(x) == abs(v1)

    def test__neq(self, v1):
        x = self.inst(v1)
        assert -x == -v1
    
    def test__pos(self, v1):
        x = self.inst(v1)
        assert +x == +v1

    def test_add(self, v1, v2):
        x = self.inst(v1)
        y = self.inst(v2)
        assert x + v2 == v1 + v2
        assert x + y == v1 + v2
        assert y + x == v1 + v2
        assert v2 + x == v1 + v2

    def test_sub(self, v1, v2):
        x = self.inst(v1)
        y = self.inst(v2)
        assert x - v2 == v1 - v2
        assert x - y == v1 - v2
        assert y - x == v2 - v1
        assert v2 - x == v2 - v1

    def test_mul(self, v1, v2):
        x = self.inst(v1)
        y = self.inst(v2)
        assert x*v2 == v1*v2
        assert x*y == v1*v2
        assert y*x == v2*v1
        assert v2*x == v2*v1

    def test_floordiv(self, v1, v2):
        x = self.inst(v1)
        y = self.inst(v2)
        assert x//v2 == v1//v2
        assert x//y == v1//v2
        assert y//x == v2//v1
        assert v2//x == v2//v1

    def test_truediv(self, v1, v2):
        x = self.inst(v1)
        y = self.inst(v2)
        assert x/v2 == v1/v2
        assert x/y == v1/v2
        assert y/x == v2/v1
        assert v2/x == v2/v1

    def test_mod(self, v1, v2):
        x = self.inst(v1)
        y = self.inst(v2)
        assert x%5 == v1%5
        with pytest.raises(Exception) as exec_info:
            x%v2
            x%y

    def test_pow(self, v1, v2):
        x = self.inst(v1)
        y = self.inst(v2)
        assert x**v2 == v1**v2
        with pytest.raises(Exception) as exec_info:
            x**y

    def test_eq(self, v1, v2):
        x = self.inst(v1)
        x2 = self.inst(v1)
        y = self.inst(v2)

        assert (x==v1)
        assert (x==x2)
        assert not(x==v2)
        assert not(x==y)

    def test_ne(self, v1, v2):
        x = self.inst(v1)
        x2 = self.inst(v1)
        y = self.inst(v2)

        assert not(x!=v1)
        assert not(x!=x2)
        assert (x!=v2)
        assert (x!=y)

    def test_less(self, v1):
        x = self.inst(v1)
        ym = self.inst(v1-1)
        yp = self.inst(v1+1)

        assert not(x<v1-1)
        assert not(x<ym)

        assert not(x<v1)
        assert not(x<x)

        assert (x<v1+1)
        assert (x<yp)

    def test_lesseq(self, v1):
        x = self.inst(v1)
        ym = self.inst(v1-1)
        yp = self.inst(v1+1)

        assert not(x<=v1-1)
        assert not(x<=ym)

        assert (x<=v1)
        assert (x<=x)

        assert (x<=v1+1)
        assert (x<=yp)

    def test_greater(self, v1):
        x = self.inst(v1)
        ym = self.inst(v1-1)
        yp = self.inst(v1+1)

        assert (x>v1-1)
        assert (x>ym)

        assert not(x>v1)
        assert not(x>x)

        assert not(x>v1+1)
        assert not(x>yp)

    def test_greatereq(self, v1):
        x = self.inst(v1)
        ym = self.inst(v1-1)
        yp = self.inst(v1+1)

        assert (x>=v1-1)
        assert (x>=ym)

        assert (x>=v1)
        assert (x>=x)

        assert not(x>=v1+1)
        assert not(x>=yp)

    def test_SetValue(self, v1, a1):
        x = self.inst(v1)
        x.SetValue(a1)

        assert (x.value == a1).all()




        
        