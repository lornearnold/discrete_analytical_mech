"""USCS (Unified Soil Classification System) classification functions."""

import numpy as np

class Simple_sample():

    def __init__(self, gravel, sand, fines, cu=0.0, cc=0.0, organics=None, LL=40, PL=50, LL_oven_dried=None):
        self.gravel = gravel
        self.sand = sand
        self.fines = fines
        self.cu = cu
        self.cc = cc
        self.organics = organics
        self.LL = LL
        self.PL = PL
        self.LL_oven_dried = LL_oven_dried if LL_oven_dried is not None else LL
        
        
        self.major = ""
        self.modifier = []
        self.minor = []
        self.group_symbol = ""
        self.group_name = ""
        
        self.soil_symbol_options = [
            "G",
            "S",
            "M",
            "C",
            "CL-ML",
            "O"
        ]
        self.mod_symbol_options = [
            "P",
            "W",
            "M",
            "C",
        ]
        self.soil_type_options = [
            "gravel",
            "sand",
            "silt",
            "clay",
            "silty clay",
            "silt (or clay)",
            ""
        ]
        self.coarse_modifiers = [
            "poorly graded",
            "well-graded",
            "silty",
            "clayey",
            "silty, clayey",
            "silty (or clayey)"
        ]
        self.fine_modifiers = [
            "gravelly",
            "sandy",
            ["", "elastic"],
            ["lean", "fat"],
            "",
            "organic",
            ""
        ]
        self.silt_modifiers = [
            "",
            "elastic",
            "gravelly",
            "sandy"
        ]
        self.clay_modifiers = [
            "lean",
            "fat",
            "gravelly",
            "sandy"
        ]

        self.PI = PL - LL

    def _build_group_name(self):
        """
        Build the group name based on the major, modifier, and minor constituents.
        """
        name = ""
        for mod in self.modifier:
            name += mod + " "
        
        name += self.major

        for i, minor in enumerate(self.minor):
            if i == 0:
                name += " with "
            else:
                name += " and "
            name += minor

        return name

# full classification
def classify_uscs(soil:Simple_sample):
    result = _checker(soil, _coarse_fine_check)
    soil.group_name = soil._build_group_name()
    return result

def _checker(soil,func):
    result = func(soil)
    if callable(result):
        return _checker(soil,result)
    else:
        return result
# 1:Coarse/fine check
def _coarse_fine_check(soil):
    # if coarse, return coarse major check function
    if (soil.gravel + soil.sand) >= soil.fines: # TODO: check greater/equal
        return _coarse_major_check
    else:
        return _fine_major_check

# 2:Major consituent
def _coarse_major_check(soil):
    if soil.gravel > soil.sand:
        i = 0
    else:
        i = 1
    soil.major = soil.soil_type_options[i]
    soil.group_symbol = soil.soil_symbol_options[i]
    return _coarse_clean_check

def _fine_major_check(soil):
    return _plasticity_check

def _coarse_clean_check(soil):
    if soil.fines > 0.12:
#         if soil.LL == None: # TODO: move to plasticity check
#             soil.modifier += soil.coarse_modifiers[5]
#             return _coarse_minor_check
#         else:
        return _plasticity_check
    else:  
        return _gradation_quality_check
            
# 4: Quality?
def _gradation_quality_check(soil):
    cu_lim = 4 * (soil.sand>soil.gravel) + 6 * (soil.gravel>soil.sand) #  1 -> sand, 0 -> gravel
        
    well = [
        soil.cu >= cu_lim,
        soil.cc >= 1,
        soil.cc <= 3
    ]
    
    if all(well):
        i = 1
    else:
        i = 0
        
    soil.modifier += [soil.coarse_modifiers[i]]
    soil.group_symbol += soil.mod_symbol_options[i]
    
    if soil.fines > 0.05:
        return _plasticity_check
    else:
        return _minor_check

# 5: Minor?
def _fines_pre_minor_check(soil):
    if soil.fines >= 0.7:
        return _minor_check
    else:
        soil.modifier += [soil.fine_modifiers[int(soil.sand>soil.gravel)]]

def _minor_check(soil):
    i = int(soil.gravel>soil.sand) # 0 if sand>gravel, otherwise, 1
    minor_coarse = soil.sand * i + soil.gravel * (1-i) 
    
    if minor_coarse >= 0.15:
        soil.minor += [soil.soil_type_options[i]]
    return None

def _plasticity_check(soil):
    # organic, silt or clay, high or low
    # organic check if soil.LL_oven_dried provided AND soil.major is fine
    # A-line check 
    
    if soil.PI >= _A_line(soil.LL):
        if soil.PI >= _A_dual_line(soil.LL):
            i = 3
        else:
            i = 4
    else:
        i = 2
    
    if soil.major == "":
        # major type not yet set -> not a coarse-grained sample
        if _is_organic(soil.LL, soil.LL_oven_dried):
            soil.modifier += [soil.fine_modifiers[5]]
            soil.group_symbol = soil.soil_symbol_options[5]
            if i == 4:
                i = 3 # ignore CL-ML zone for organic soil
        else:
            soil.group_symbol = soil.soil_symbol_options[i]
        soil.major = soil.soil_type_options[i]
        return _fines_pre_minor_check
    
    # major type already set -> coarse-grained sample
    if soil.fines >= 0.12:
        soil.modifier += [soil.coarse_modifiers[i]]
        soil.group_symbol += soil.soil_symbol_options[i]
    else:
        soil.minor = [soil.soil_type_options[i]]
        soil.group_symbol += "-" + soil.group_symbol[0] + soil.soil_symbol_options[i]
    
    return _minor_check
    
def _A_line(LL, L0=4, slope=0.73, offset=20):
    L1 = L0/slope + offset
    return np.piecewise(
        LL,
        [LL <= L0, L0 < LL <= L1, LL > L1],
        [LL, L0, slope * (LL - offset)]
    )

def _A_dual_line(LL):
    return _A_line(LL, L0=7)

def _is_organic(LL, LL_oven_dried): 
        return (LL_oven_dried / LL) < 0.75
    
if __name__ == "__main__":
    # Example usage
    sample = Simple_sample(gravel=0.38, sand=0.51, fines=0.11, cu=3, cc=1.2)
    classification = classify_uscs(sample)
    print(f"Major: {sample.major}, Modifier: {sample.modifier}, Minor: {sample.minor}, Group Symbol: {sample.group_symbol}, Group Name: {sample.group_name}")