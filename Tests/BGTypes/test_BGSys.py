from GEFF.bgtypes import BGVal, BGFunc, BGSystem, Val, Func
import pytest

class TestBGSystem():
    @pytest.fixture
    def sample_bgval(self):
        x = BGVal("x", 2, 1)
        return x
    
    @pytest.fixture
    def sample_bgfunc(self, sample_bgval):
        f = BGFunc("f", [sample_bgval], 2, 1)
        return f
    
    def init(self):
        x = BGVal("x", 2, 1)
        f = BGFunc("f", [x], 2, 1)
        sys = BGSystem({x, f}, 0.55, 0.32)
        return sys
    
    def test_init(self):
        U = self.init()
        assert U.H0 == 0.55
        assert U.MP == 0.32

    def test_initFromU(self):
        U = self.init()
        U.initialise("x")(10)

        #check full copy
        V = BGSystem.from_system(U, copy=True)

        assert V.H0 == U.H0
        assert V.MP == U.MP
        assert V.quantity_names() == U.quantity_names()
        assert V.value_list() == U.value_list()
        assert V.function_list() == U.function_list()

        #check empty copy
        V = BGSystem.from_system(U, copy=False)

        assert V.H0 == U.H0
        assert V.MP == U.MP
        assert V.quantity_names() == U.quantity_names()
        assert V.value_list() == []
        assert V.function_list() == []

    def test_quantity_set_and_names(self):
        U = self.init()
        for val in ["x", "f"]:
            assert val in U.quantity_names()

    def test_initialise_value(self):
        U = self.init()
        U.initialise("x")(10)
        assert isinstance(U.x, Val)
        assert U.x.value == 10
        assert U.x.get_conversion() == 0.55**2 * 0.32

    def test_initialise_func(self):
        U = self.init()
        def func(x): return 5
        U.initialise("f")(func)
        assert isinstance(U.f, Func)
        assert callable(U.f)
        assert U.f.get_basefunc()(0.3421) == func(0.342) 
        assert U.f.get_conversion() == 0.55**2 * 0.32
    
    def test_initialise_unknown(self):
        U = self.init()
        with pytest.raises(Exception):
            U.initialise("a")(10)

    
    def test_value_list_and_names(self):
        U = self.init()
        assert U.value_list() == []
        U.initialise("x")(10)
        assert "x" in U.value_names()
        U.initialise("f")(lambda x: 5)
        assert "f" not in U.value_names()

    def test_function_list_and_names(self):
        U = self.init()
        assert U.function_list() == []
        U.initialise("x")(10)
        assert "x" not in U.function_names()
        U.initialise("f")(lambda x: 5)
        assert "f" in U.function_names()

    def test_remove(self):
        U = self.init()
        U.initialise("x")(10)
        U.remove("x")
        
        assert not(hasattr(U, "x"))
        assert "x" not in U.quantity_names()

    def test_add_obj(self):
        U = self.init()
        U.add_BGVal("y", 0, 2)
        U.add_BGFunc("g", [U.objects["y"]], 2, 0)
        names = U.quantity_names()
        assert "y" in names
        assert "g" in names

    """def test_AddVal(self):
        U = self.createSys()
        U.AddValue("y", 3.0, 0, 2)
        U.AddFunction("g", [U._y], lambda x: x, 2, 0)
        names = U.quantity_names()
        assert "y" in names
        assert "g" in names"""


    def test_set_units_false(self):
        U = self.init()
        U.initialise("x")(10)
        U.initialise("f")(lambda x: 5)

        U.set_units(False)

        assert not(U.x.get_units())
        assert not(U.f.get_units())
    
    def test_set_unitsTrue(self):
        U = self.init()

        U.initialise("x")(10)
        U.initialise("f")(lambda x: 5)

        U.set_units(True)

        assert U.x.get_units()
        assert U.f.get_units()
    

        



