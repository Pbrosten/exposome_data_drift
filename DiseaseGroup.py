import HelperFunctions as Helper

def SkinCancer(kind='all'):
    '''
    This function generates the ICD-10 codes for the skin cancer disease group.

    Inupt:
        kind <- str; {'all': produces all neoplasms of skin,
                    'malignant': restricts only to malignant neoplasms of skin,
                    'benign': restricts only to benign neoplasms of skin}
    Output:
        SkinCodes <- list; list consisting of ICD-10 codes corresponding to the desired group
    '''
    
    BenSkinDict = {'23': [0,1,2,3,4,5,6,7,9]}   # Other benign neoplasms of skin

    if kind == 'malignant':
        MalSkinDict = {'43': [0,1,2,3,4,5,6,7,8,9],    # Malignant melanoma of skin
            '44': [0,1,2,3,4,5,6,7,8,9]}    # Other malignant neoplasms of skin
        
        SkinCodes = Helper.ICDCodes(letter='C', dict=MalSkinDict)
        return SkinCodes

    elif kind == 'benign':
        BenSkinDict = {'23': [0,1,2,3,4,5,6,7,9]}   # Other benign neoplasms of skin

        SkinCodes = Helper.ICDCodes(letter='D', dict=BenSkinDict)
        return SkinCodes

    elif kind == 'all':
        MalSkinDict = {'43': [0,1,2,3,4,5,6,7,8,9],    # Malignant melanoma of skin
            '44': [0,1,2,3,4,5,6,7,8,9]}    # Other malignant neoplasms of skin
        BenSkinDict = {'23': [0,1,2,3,4,5,6,7,9]}   # Other benign neoplasms of skin
        
        MalSkinCodes = Helper.ICDCodes(letter='C', dict=MalSkinDict)
        BenSkinCodes = Helper.ICDCodes(letter='D', dict=BenSkinDict)
        SkinCodes = MalSkinCodes + BenSkinCodes
        return SkinCodes

def CVD(kind='base'):
    '''
    This function generates the ICD-10 codes for the cardio vascular disease group.

    Inupt:
        kind <- str; {'base': produces whole CVD group,
                    'hyper': produces only hypertension ICD-10 codes}
    Output:
        CVDCodes <- list; list consisting of ICD-10 codes corresponding to the desired group 
    '''
    if kind == 'base':
        CVDDict = {'20': [0,1,8,9],                 # Ischaemic heart diseases
            '21': [0,1,2,3,4,9],
            '22': [0,1,8,9],
            '23': [0,1,2,3,5,6,8],
            '24': [0,1,8,9],
            '25': [0,1,2,3,4,5,6,8,9],
            '50': [0,1,9],                  # Heart failure
            '60': [0,1,2,3,4,5,6,7,8,9],    # Cerebrovascular diseases
            '61': [0,1,2,3,4,5,6,8,9],
            '62': [0,1,9],
            '63': [0,1,2,3,4,5,6,8,9],
            '64': False,
            '65': [0,1,2,3,8,9],
            '66': [0,1,2,3,4,8,9],
            '67': [0,1,2,3,4,5,6,7,8,9],
            '68': [0,8],
            '69': [0,1,2,3,4,8]}

        CVDCodes = Helper.ICDCodes(letter='I', dict=CVDDict) + ['F010','F011','F012','F013','F018','F019']
        return CVDCodes

    elif kind == 'hyper':
        HyperDict = {'10': False,                  # Hypertension
                '11': [0,9],
                '12': [0,9],
                '13': [0,1,2,9],
                '15': [0,1,2,8,9]}
        HyperCodes = Helper.ICDCodes(letter='I', dict=HyperDict)
        return HyperCodes

def BreastCancer():
    '''
    This function generates the ICD-10 codes for the breast cancer disease group.

    Input:
        None
    Outout:
        BreastCodes <- list; list consisting of ICD-10 codes corresponding to the desired group
    '''
    BreastDict = {'50': [0,1,2,3,4,5,6,8,9]}    # Malignant neoplasm of breast
    BreastCodes = Helper.ICDCodes(letter='C', dict=BreastDict)
    return BreastCodes

def LungCancer():
    '''
    This function generates the ICD-10 codes for the lung cancer disease group.

    Input:
        None
    Outout:
        LungCodes <- list; list consisting of ICD-10 codes corresponding to the desired group
    '''
    LungDict = {'34': [0,1,2,3,8,9]}            # Malignant neoplasm of bronchus and lung    
    LungCodes = Helper.ICDCodes(letter='C', dict=LungDict)
    return LungCodes

def ProstateCancer():
    '''
    This function generates the ICD-10 codes for the prostate cancer disease group.

    Input:
        None
    Outout:
        ProstateCodes <- list; list consisting of ICD-10 codes corresponding to the desired group
    '''
    ProstateDict = {'61': False}                # Malignant neoplasm of prostate    
    ProstateCodes = Helper.ICDCodes(letter='C', dict=ProstateDict)
    return ProstateCodes

def Diabetes():
    '''
    This function generates the ICD-10 codes for the diabetes disease group.

    Input:
        None
    Outout:
        DiabetesCodes <- list; list consisting of ICD-10 codes corresponding to the desired group
    '''
    DiabetesDict = {'10': [0,1,2,3,4,5,6,7,8,9],# Insulin-dependent diabetes mellitus
                '11': [0,1,2,3,4,5,6,7,8,9],# Non-insulin-dependent diabetes mellitus
                '12': [1,3,5,8,9],          # Malnutrition-related diabetes mellitus
                '13': [0,1,2,3,4,5,6,7,8,9],# Other specified diabetes mellitus
                '14': [0,1,2,3,4,5,6,7,8,9]}# Unspecified diabetes mellitus   
    DiabetesCodes = Helper.ICDCodes(letter='E', dict=DiabetesDict)
    return DiabetesCodes

