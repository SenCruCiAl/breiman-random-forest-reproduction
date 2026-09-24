class Microwave:
    def __init__(self, brand:str , power_rating: str)->None:
        self.brand = brand
        self.power_rating = power_rating  




LG: Microwave = Microwave('LG','B')
print(LG.brand)
print(LG.power_rating)
print(LG)

bosch: Microwave= Microwave("Bosche", "A")
print(bosch.brand)
print(bosch.power_rating)
print(bosch)

