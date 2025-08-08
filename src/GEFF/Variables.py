from GEFF.BGTypes import BGVal

        
class ScalarField:
    def __init__(self, name, KG):
        self.name = name
        names = [name, f"d{name}"]
        self.__KG = KG

    def CreateQuantities(self):
        field = BGVal(self.name, 0, 1)
        dfield = BGVal(f"d{self.name}", 1, 1)
        return {field, dfield}
        
class GaugeField:
    def __init__(self, name, EoM):
        pass




            

            


    


