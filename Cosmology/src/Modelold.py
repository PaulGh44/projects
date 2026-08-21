# This class defines the model = background. The equation of motion of the Riccati variable is completely determined by the knoweledge of the scale factor along the complex contour. The parameters of the model are
# - The type of asymptotics in the Euclidean region (either EAdS or EdS)
# - In particular wineglass models such as the one in https://arxiv.org/abs/2602.23432, the scale factor was defined piecewise. We thus include parameters that indicates the the interfaces in Euclidean and Lorentzian regions. For example, $\tau_{min} would be include in list_params_euclidean and list_params_lorentzian would be empty for the background described in https://arxiv.org/abs/2602.23432.
# - We also include the list of parameters in both region. The first parameter is always the Hubble constant of each region.


class model:
    def __init__(self, model_name:str, Asymptotics:str,list_params_euclidean: list[float], list_params_lorentzian: list[float], list_euclidean_interfaces: list[float], list_lorentzian_interfaces: list[float]):

        self.model_name = model_name
        self.Asymptotics = Asymptotics
        self.list_params_euclidean = list_params_euclidean
        self.list_params_lorentzian = list_params_lorentzian
        self.list_lorentzian_interfaces = list_lorentzian_interfaces
        self.list_euclidean_interfaces = list_euclidean_interfaces
        self.scale_factor = None
        self.scale_factor_derivative = None
    

    def get_model_name(self):
        return self.model_name

    def get_Asymptotics(self):
        return self.Asymptotics
    
    def get_list_params_euclidean(self):
        return self.list_params_euclidean

    def get_list_params_lorentzian(self):
        return self.list_params_lorentzian
    
    def get_list_params_lorentzian(self):
        return self.list_params_lorentzian
    
    def get_list_euclidean_interfaces(self):
        return self.list_euclidean_interfaces
    
    def get_list_lorentzian_interfaces(self):
        return self.list_lorentzian_interfaces
    
    def get_Number_euclidean_regions(self):
        return len(self.list_euclidean_interfaces)
    
    def get_Number_lorentzian_regions(self):
        return len(self.list_lorentzian_interfaces)
    
    def get_scale_factor(self):
        return self.scale_factor
    
    def get_scale_factor_derivative(self):
        self.scale_factor_derivative
    
    def set_scale_factor(self,scale_factor):
        self.scale_factor = scale_factor

    def set_scale_factor_derivative(self, scale_factor_derivative):
        self.scale_factor_derivative = scale_factor_derivative