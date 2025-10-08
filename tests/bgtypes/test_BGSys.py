from geff.bgtypes import BGVar, BGConst, BGFunc, BGSystem, Val, Func
import pytest

class TestBGSystem():
    @pytest.fixture
    def sample_bgval(self):
        x = BGVar("x", 2, 1)
        return x
    
    @pytest.fixture
    def sample_bgfunc(self, sample_bgval):
        f = BGFunc("f", [sample_bgval], 2, 1)
        return f
    
    def init(self):
        x = BGVar("x", 2, 1)
        c = BGConst("c", 0, 2)
        f = BGFunc("f", [x], 2, 1)
        sys = BGSystem({x, c, f}, 0.55, 0.32)
        return sys
    
    def test_init(self):
        U = self.init()
        assert U.omega == 0.55
        assert U.mu == 0.32

    def test_initFromU(self):
        U = self.init()
        U.initialise("x")(10)

        #check full copy
        V = BGSystem.from_system(U, copy=True)

        assert V.omega == U.omega
        assert V.mu == U.mu
        assert V.quantity_names() == U.quantity_names()
        assert V.variable_names() == U.variable_names()
        assert V.function_names() == U.function_names()
        assert V.constant_names() == U.constant_names()

        V.units = False

        assert not(U.units) == V.units

        assert V.x.value == U.x.value*U.x.conversion**(-U.x.units)

        #check empty copy
        V = BGSystem.from_system(U, copy=False)

        assert V.omega == U.omega
        assert V.mu == U.mu
        assert V.quantity_names() == U.quantity_names()
        assert V.variable_list() == []
        assert V.function_list() == []
        assert V.constant_list() == []

    def test_quantity_set_and_names(self):
        U = self.init()
        for val in ["x", "f"]:
            assert val in U.quantity_names()

    def test_initialise_value(self):
        U = self.init()
        U.initialise("x")(10)
        assert isinstance(U.x, Val)
        assert U.x.value == 10
        assert U.x.conversion == 0.55**2 * 0.32

    def test_initialise_func(self):
        U = self.init()
        def func(x): return 5
        U.initialise("f")(func)
        assert isinstance(U.f, Func)
        assert callable(U.f)
        assert U.f.basefunc(0.3421) == func(0.342) 
        assert U.f.conversion == 0.55**2 * 0.32
    
    def test_initialise_unknown(self):
        U = self.init()
        with pytest.raises(Exception):
            U.initialise("a")(10)
    
    def test_value_list_and_names(self):
        U = self.init()
        assert U.variable_list() == []
        U.initialise("x")(10)
        assert "x" in U.variable_names()
        U.initialise("c")(0.5)
        assert "c" not in U.variable_names()
        U.initialise("f")(lambda x: 5)
        assert "f" not in U.variable_names()

    def test_const_list_and_names(self):
        U = self.init()
        assert U.constant_list() == []
        U.initialise("x")(10)
        assert "x" not in U.constant_names()
        U.initialise("c")(0.5)
        assert "c" in U.constant_names()
        U.initialise("f")(lambda x: 5)
        assert "f" not in U.constant_names()

    def test_function_list_and_names(self):
        U = self.init()
        assert U.function_list() == []
        U.initialise("x")(10)
        assert "x" not in U.function_names()
        U.initialise("c")(0.5)
        assert "c" not in U.function_names()
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
        U.add_variable("y", 0, 2)
        U.add_constant("d", 2, 1)
        U.add_function("g", [U.quantities["y"]], 2, 0)
        names = U.quantity_names()
        assert "y" in names
        assert "g" in names
        assert "d" in names

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

        U.units = False

        assert not(U.x.units)
        assert not(U.f.units)
    
    def test_set_unitsTrue(self):
        U = self.init()

        U.initialise("x")(10)
        U.initialise("f")(lambda x: 5)

        U.units = True

        assert U.x.units
        assert U.f.units
    

        



