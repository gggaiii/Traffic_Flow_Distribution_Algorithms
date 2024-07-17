class District:
    def __init__(self, name, data, population, pop_density, economic, poi_num):
        self.name = name
        self.data = data
        self.population = population
        self.pop_density = pop_density
        self.economic = economic
        self.poi_num = poi_num
        self.comqz = self.calculate_comqz()

    def get_name(self):
        return self.name

    def get_data(self):
        return self.data

    def get_population(self):
        return self.population

    def get_pop_density(self):
        return self.pop_density

    def get_economic(self):
        return self.economic

    def get_poi_num(self):
        return self.poi_num

    def get_comqz(self):
        return self.comqz

    def set_name(self):
        return self.name

    def set_data(self, data):
        self.data = data

    def set_population(self, population):
        self.population = population
        self.comqz = self.calculate_comqz()

    def set_pop_density(self, pop_density):
        self.pop_density = pop_density
        self.comqz = self.calculate_comqz()

    def set_economic(self, economic):
        self.economic = economic
        self.comqz = self.calculate_comqz()

    def set_poi_num(self, poi_num):
        self.poi_num = poi_num
        self.comqz = self.calculate_comqz()

    def calculate_comqz(self):
        return self.population * self.pop_density * self.economic * self.poi_num