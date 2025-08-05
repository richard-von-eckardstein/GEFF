from src.BGQuantities.BGTypes import BGVal, BGFunc, BGSystem, Val, Func
import pytest
import random

class TestBGSystem():
    @pytest.fixture
    def sample_bgval(self):
        x = BGVal("x", 2, 1)
        return x
    
    @pytest.fixture
    def sample_bgfunc(self, sample_bgval):
        f = BGFunc("f", [sample_bgval], 2, 1)
        return f
    
    def createSys(self):
        x = BGVal("x", 2, 1)
        f = BGFunc("f", [x], 2, 1)
        sys = BGSystem([x, f])
        return sys
    
    def init(self):
        sys = self.createSys()
        U = sys(0.55, 0.32)
        return U
    
    def test_init(self):
        U = self.init()
        assert U.H0 == 0.55
        assert U.MP == 0.32

    def test_initFromU(self):
        U = self.init()
        U.Initialise("x")(10)

        sys = self.createSys()
        V = sys.FromSystem(U)

        assert V.H0 == U.H0
        assert V.MP == U.MP
        assert V.ObjectNames() == U.ObjectNames()
        assert V.ValueList() == U.ValueList()

    def test_initFromDic(self):
        U = self.init()
        U.Initialise("x")(10)
        U.Initialise("f")(lambda x: 5)
        sys = self.createSys()

        dic = {"f":lambda x: 5}
        V = sys.FromDic(dic, U.H0, U.MP)

        assert V.H0 == U.H0
        assert V.MP == U.MP
        assert V.ObjectNames() == U.ObjectNames()
        assert V.ValueList() == []

    def test_Object_SetAndNames(self, sample_bgfunc, sample_bgval):
        sys = self.createSys()
        for val in ["x", "f"]:
            assert val in sys.ObjectNames()

    def test_InitialiseValue(self):
        U = self.init()
        U.Initialise("x")(10)
        assert isinstance(U.x, Val)
        assert U.x.value == 10
        assert U.x.GetConversion() == 0.55**2 * 0.32

    def test_InitialiseFunc(self):
        U = self.init()
        func = lambda x: 5
        U.Initialise("f")(func)
        assert isinstance(U.f, Func)
        assert callable(U.f)
        randval = random.random()
        assert U.f.GetBaseFunc()(randval) == func(randval) 
        assert U.f.GetConversion() == 0.55**2 * 0.32
    
    def test_InitialiseUnknown(self):
        U = self.init()
        with pytest.raises(Exception) as exc_info:
            U.Initialise("a")(10)

    
    def test_Value_ListAndNames(self):
        U = self.init()
        assert U.ValueList() == []
        U.Initialise("x")(10)
        assert "x" in U.ValueNames()
        U.Initialise("f")(lambda x: 5)
        assert "f" not in U.ValueNames()

    def test_Function_ListandNames(self):
        U = self.init()
        assert U.FunctionList() == []
        U.Initialise("x")(10)
        assert "x" not in U.FunctionNames()
        U.Initialise("f")(lambda x: 5)
        assert "f" in U.FunctionNames()

    def test_Remove(self):
        U = self.init()
        U.Initialise("x")(10)
        U.Remove("x")
        
        assert not(hasattr(U, "x"))

    """
    def test_AddObj(self):
        U = self.createSys()
        U.AddBGVal("y", 0, 2)
        U.AddBGFunc("g", [U._y], 2, 0)
        names = U.ObjectNames()
        assert "y" in names
        assert "g" in names

    def test_AddVal(self):
        U = self.createSys()
        U.AddValue("y", 3.0, 0, 2)
        U.AddFunction("g", [U._y], lambda x: x, 2, 0)
        names = U.ObjectNames()
        assert "y" in names
        assert "g" in names
    """


    def test_SetUnitsFalse(self):
        U = self.init()
        U.Initialise("x")(10)
        U.Initialise("f")(lambda x: 5)

        U.SetUnits(False)

        assert U.x.GetUnits() == False
        assert U.f.GetUnits() == False
    
    def test_SetUnitsTrue(self):
        U = self.init()

        U.Initialise("x")(10)
        U.Initialise("f")(lambda x: 5)

        U.SetUnits(True)

        assert U.x.GetUnits() == True
        assert U.f.GetUnits() == True
    

        



