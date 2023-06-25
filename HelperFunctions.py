from pandas import to_datetime, concat
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from numpy import delete, array, where, argmax, nan_to_num
from sklearn.metrics import precision_recall_curve
'''
Functions to help consolidate unnessecary clutter in EDA notebook.
'''

def FilterDate(df, cutoff):
    '''
    This function filters out any rows which include dates (YYYY-MM-DD)
    below a specified threshold.

    Input:
        df <- DataFrame of dates in string format (YYYY-MM-DD)
        cutoff <- date threshold in string format
    Output:
        filtered DataFrame

    '''
    Early = (df < cutoff)
    return df[~Early].dropna(axis=1, how='all').dropna(axis=0, how='all')

def ICDCodes(letter, dict):
    '''
    This function constructs a list of ICD-10 codes from a dictionary of these codes.
    
    Input:
        letter <- string character; represents begining of ICD-10 code
        dict <- dictionary; contains all numerical diagnoses
    Output:
        listICD <- list; contains all relavent ICD-10 codes
    '''
    listICD = []
    for pre in dict.keys():
        if not dict[pre]:
            listICD.append(f'{letter}{pre}')
            continue
        for post in dict[pre]:
            listICD.append(f'{letter}{pre}{post}')
    return listICD

def positiveClass(df, ICDs, colRange):
    '''
    This function collects all f.eid values for patients having being one of the given ICD-10
    codes. This collection is a set to ensure uniqueness of entries.

    Input:
        df <- DataFrame; contains all diagnosis codes for each patient
        ICDs <- list; all ICD-10 codes to be considered
        colRange <- list; number of columns (visits) to be looked at
    Output:
        positiveClass <- set; contains all f.eid values of patients in the positive class
    '''
    positiveClass = set()
    for i in colRange:
        positiveClass.update(df.loc[df[f'f.41270.0.{i}'].isin(ICDs)]['f.eid'].values)
    return positiveClass

def negativeClass(df, ICDs, colRange):
    '''
    This function collects all f.eid values for patients never having recieved one of the given ICD-10
    codes. This collection is a set to ensure uniqueness of entries.

    Input:
        df <- DataFrame; contains all diagnosis codes for each patient
        ICDs <- list; all ICD-10 codes to be ignored
        colRange <- list; number of columns (visits) to be looked at
    Output:
        positiveClass <- set; contains all f.eid values of patients in the positive class
    '''
    negativeClass = set()
    for i in colRange:
        negativeClass.update(df.loc[~df[f'f.41270.0.{i}'].isin(ICDs)]['f.eid'].values)
    return negativeClass

def dfRelComp(df1, df2):
    '''
    This function takes the relative complement of two DataFrames by removing from df1 all
    rows shared with df2.

    Input:
        df1 <- DataFrame; primary DataFrame
        df2 <- DataFrame; secondary DataFrame
    Output:
        df_complement <- DataFrame; equivalent to df1 \ df2
    '''
    return df1[~df1.index.isin(df2.index)]

def positiveCohortSplit(ICDs, positiveTrain, df_codes, df_dates, df_Main, positiveVal,
                        cutoff=False, exclusive=True):
    '''
    This function pulls all participants who have been diagnosed with from a set of 
    ICD-10 codes. The purpose is to construct two DataFrames of participants which 
    satisfy the requirements for group membership in each. Mainly, healthy at visit 0
    and diagnosed by visit 1 (Cohort 1); healthy through visit 1 and diagnosed after
    (Cohort 2)

    Input:
        ICDs <- set; all ICD-10 codes under consideration
        positiveTrain <- set; f.eid values of participants to be used for training
        df_codes <- DataFrame; all diagnosis codes
        df_dates <- DataFrame; all diagnosis dates
        df_Main <- DataFrame; all participants for both train and validation
        positiveVal <- set; f.eid values of participants to be used for traceability validation
        cutoff <- boolean;  
                    if (True): enforce the given cutoff period
        exclusive <- boolean;
                    if (False): test washed validation positives for training membership
    Output:
        CohortTrain <- DataFrame; all positive cohort 1 members
        CohortVal <- DataFrame; all positive cohort 2 members
        washDict <- dict; dictionary of washed participants at each stage
    '''

    # All participants with a diagnosis of interest (All codes included)
    df_posTrain = df_codes[df_codes['f.eid'].isin(positiveTrain)].drop(['Iid'], axis=1).set_index('f.eid')
    df_posVal = df_codes[df_codes['f.eid'].isin(positiveVal)].drop(['Iid'], axis=1).set_index('f.eid')

    # Patients versus visit with entries as dates
    ## (Mirror of df_pos with ICD-10 codes replaced by date of diagnosis)
    posTrainDates = df_dates[df_dates['f.eid'].isin(positiveTrain)].drop(['Iid'], axis=1).set_index('f.eid')
    posTrainDates = posTrainDates.rename(columns = lambda col: f'{col[:5]}70{col[-4:]}')
    posValDates = df_dates[df_dates['f.eid'].isin(positiveVal)].drop(['Iid'], axis=1).set_index('f.eid')
    posValDates = posValDates.rename(columns = lambda col: f'{col[:5]}70{col[-4:]}')
    
    # Strip all ICD-10 diagnoses not of interest and remove empty columns
    df_posTrain = df_posTrain.where(df_posTrain.isin(ICDs)).dropna(axis=1, how = 'all')
    df_posVal = df_posVal.where(df_posVal.isin(ICDs)).dropna(axis=1, how = 'all')

    # All participants in each set who recieved an ICD-10 diagnosis of interest
    df_posTrain = df_posTrain.where(cond=df_posTrain.isna(), other=posTrainDates).dropna(axis=0, how='all').dropna(axis=1, how='all')
    df_posVal = df_posVal.where(cond=df_posVal.isna(), other=posValDates).dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    # Convert to datetime
    for col in df_posTrain:
        df_posTrain[col] = to_datetime(df_posTrain[col])
    for col in df_posVal:
        df_posVal[col] = to_datetime(df_posVal[col])
        

    df_Train = df_Main.merge(df_posTrain.min(axis=1).rename('Earliest'), left_index=True, right_index=True)
    df_Val = df_Main.merge(df_posVal.min(axis=1).rename('Earliest'), left_index=True, right_index=True)
        
    #if exclusive:
    #    df_Train = dfRelComp(df_Train, df_Val)

    # Compute number of washed participants from each stage and cohort membership
    washDict = dict()

    # Wash participants with pre-existing diagnoses at assessment
    prewash = df_Train.shape[0] + df_Val.shape[0]
    df_Train = df_Train[df_Train['Earliest'] >= df_Train['Visit 0']]
    prewash -= df_Train.shape[0]

    # Wash train participants diagnosed between visit
    trainwash = df_Train.shape[0]
    df_Train = df_Train[df_Train['Earliest'] >= df_Train['Wash 0']]

    # if exclusive -> washed val participants are cut from the study
    if exclusive:
        df_Val = df_Val[df_Val['Earliest'] >= df_Val['Visit 1']]
        prewash -= df_Val.shape[0]
        washDict['pre'] = prewash

        valwash = df_Val.shape[0]
        df_Val = df_Val[df_Val['Earliest'] >= df_Val['Wash 1']]
        trainwash -= df_Train.shape[0]
        valwash -= df_Val.shape[0]

    # if not exclusive -> washed val participants are moved to training
    if not exclusive:
        df_Val = df_Val[df_Val['Earliest'] >= df_Val['Visit 0']]
        prewash -= df_Val.shape[0]
        
        Val2Train = df_Val[~(df_Val['Earliest'] >= df_Val['Wash 1'])]
        moverwash = Val2Train.shape[0]
        Val2Train = Val2Train[(Val2Train['Earliest'] >= Val2Train['Wash 0'])]
        moverwash -= Val2Train.shape[0]
        prewash -= moverwash
        washDict['pre'] = prewash

        df_Train = concat([df_Train, Val2Train])

        df_Val = df_Val[df_Val['Earliest'] >= df_Val['Wash 1']]
        trainwash = trainwash - df_Train.shape[0]
        df_Train = dfRelComp(df_Train, df_Val)
        valwash = 0

    washDict['training wash'] = trainwash
    washDict['validation wash'] = valwash

    # Filter participants diagnosed in desired time range
    if cutoff:
        cutoffwash = df_Train.shape[0] + df_Val.shape[0]
        cohortTrain = df_Train[df_Train['Earliest'] < df_Train['Cutoff 0']]
        cohortVal = df_Val[df_Val['Earliest'] < df_Val['Cutoff 1']]

        cutoffwash = cutoffwash - cohortTrain.shape[0] - cohortVal.shape[0]
        washDict['cutoff wash'] = cutoffwash
            
    else: 
        cohortTrain = df_Train.copy()
        cohortVal = df_Val.copy()

    # Calculate total number of washed participants
    totalWash = 0
    for key in washDict.keys():
        totalWash += washDict[key]

    washDict['total'] = totalWash
    
    if not exclusive:
        return cohortTrain, set(Val2Train.index), cohortVal, washDict
    else:
        return cohortTrain, cohortVal, washDict


def PSMbalancer(data, target='target', ratio=1):
    '''
    This function utilizes Propensity Score Matching (PSM) to balance the provided dataset.
    Propensity scores are constructed using all data other than target.
    Input:
        data <- DataFrame; unbalanced dataset which will be balanced
        target <- str; name of the target variable in data
        ratio <- int; number of matched majority per minority 
    Output:
        master_pairs <- dict; dictionary with keys as the treated eids and values a list of 
                            matched control eids
    '''
    # Split the data into treated and control groups
    treated = data.loc[data[target] == 1]
    control = data.loc[data[target] == 0]
    controlBatch = control.copy()

    # Fit a logistic regression model to estimate propensity scores
    X = data.drop([target], axis=1)
    y = data[target]
    logit = LogisticRegression()
    logit.fit(X, y)
    propensity_scores = logit.predict_proba(X)[:, 1]
    data['propensity_scores'] = propensity_scores
    treated = data.loc[data[target] == 1].drop(['f9', 'Sex', 'target'], axis=1)
    control = data.loc[data[target] == 0].drop(['f9', 'Sex', 'target'], axis=1)

    # Restrict matching to control
    propensity_scores = control['propensity_scores'].values
    # Enforce non-replacement of matched controls
    matchedID = []
    dropout = dict()

    treated_eid = treated.index
    master_pairs = dict.fromkeys(treated_eid)
    
    # Find nearest neighbors of treated observations based on propensity scores
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(propensity_scores.reshape(-1, 1))

    for i in range(ratio):
        fold_pairs = dict.fromkeys(treated_eid)
        for j,unmatched in enumerate(treated['propensity_scores'].values):
            match = nn.kneighbors(unmatched.reshape(1,-1), return_distance=False).flatten()[0]
            eid = controlBatch.iloc[match].name

            if eid in matchedID:
                # Remove all already matched controls
                controlBatch = dfRelComp(controlBatch, controlBatch[controlBatch.index.isin(list(dropout.values()))])
                propensity_scores = delete(arr=propensity_scores, obj=list(dropout.keys()))
                dropout.clear()
                # Refit nn
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(propensity_scores.reshape(-1,1))
                # Find new nearest neighbor
                match = nn.kneighbors(unmatched.reshape(1,-1), return_distance=False).flatten()[0]
                eid = controlBatch.iloc[match].name
                # Record propensity matched eids
                matchedID.append(eid)
                dropout[match] = eid
                fold_pairs[treated_eid[j]] = eid
            else:
                # Record propensity matched eids
                matchedID.append(eid)
                dropout[match] = eid
                fold_pairs[treated_eid[j]] = eid

        if i == 0:
            for key, value in fold_pairs.items():
                master_pairs[key] = [value]
            #master_pairs = fold_pairs.copy()
        else:
            # Add this fold's matches into the master matches list
            for key, value in fold_pairs.items():
                master_pairs[key].append(value)

        matched_controls = control[control.index == matchedID[0]]
        for ID in matchedID[1:]:
            matched_controls = concat([matched_controls, control[control.index == ID]], axis=0)

    return master_pairs

def thresholdSelector(model, val, val_labels, score_func):
    '''
    This function selects the optimal prediction threshold for a given model's.
    
    thresholdSelector seeks to maximize score_func.
    
    Input:
        model       <-sklearn model; model to be evaluated
        val         <- array; validation data
        val_labels  <- array; holds all ground truths for validation set
        score_func  <- function; function to be optimized
    
    Output:
        best_thresh <- float; optimal prediction threshold
    '''

    probs = model.predict_proba(val)[:,1]
    prec, recall, thresholds = precision_recall_curve(val_labels, probs)
    fscore = (2 * prec * recall) / (prec + recall)
    best_thresh = thresholds[argmax(nan_to_num(fscore,nan=0.0))]
    return best_thresh

def thresholdPredictor(model, X, ground, threshold, score_func):
    '''
    This function evaluates the given model performance using a specified  
    prediction threshold.
    
    the specific metric is defined by score_func parameter
    
    Input:
        model       <-sklearn model; model to be evaluated
        X           <- array; data to make prediction on
        ground      <- array; ground truth for predictions
        threshold   <- float; prediction confidence threshold
        score_func  <- function; function to be optimized
    
    Output:
        score       <- float; threshold specific score
    '''

    probs = model.predict_proba(X)
    probs = array([guess[1] for guess in probs])
    pred = where(probs > threshold, 1, 0)

    score = score_func(y_true=ground, y_pred=pred)
    return score
