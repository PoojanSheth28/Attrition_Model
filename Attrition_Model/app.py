from flask import Flask, render_template,request
import joblib

app=Flask(__name__)

@app.route('/')
def base():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    model=joblib.load('attrition_95.pkl')
    print('Model Loaded')

    mapping_dict=joblib.load('mapping_dict.pkl')
    
    
    Age=request.form.get('Age')
    BusinessTravel=request.form.get('BusinessTravel')
    DailyRate=request.form.get('DailyRate')
    Department=request.form.get('Department')
    DistanceFromHome=request.form.get('DistanceFromHome')
    Education=request.form.get('Education')
    EducationField=request.form.get('EducationField')
    EnvironmentSatisfaction=request.form.get('EnvironmentSatisfaction')
    Gender=request.form.get('Gender')
    HourlyRate=request.form.get('HourlyRate')
    JobInvolvement=request.form.get('JobInvolvement')
    JobRole=request.form.get('JobRole')
    JobSatisfaction=request.form.get('JobSatisfaction')
    MaritalStatus=request.form.get('MaritalStatus')
    MonthlyIncome=request.form.get('MonthlyIncome')
    MonthlyRate=request.form.get('MonthlyRate')
    NumCompaniesWorked=request.form.get('NumCompaniesWorked')
    OverTime=request.form.get('OverTime')
    PercentSalaryHike=request.form.get('PercentSalaryHike')
    RelationshipSatisfaction=request.form.get('RelationshipSatisfaction')
    StockOptionLevel=request.form.get('StockOptionLevel')
    TotalWorkingYears=request.form.get('TotalWorkingYears')
    TrainingTimesLastYear=request.form.get('TrainingTimesLastYear')
    WorkLifeBalance=request.form.get('WorkLifeBalance')
    YearsAtCompany=request.form.get('YearsAtCompany')
    YearsInCurrentRole=request.form.get('YearsInCurrentRole')
    YearsSinceLastPromotion=request.form.get('YearsSinceLastPromotion')
    YearsWithCurrManager=request.form.get('YearsWithCurrManager')

    BusinessTravel=mapping_dict['BusinessTravel'][BusinessTravel]
    Department=mapping_dict['Department'][Department]
    EducationField=mapping_dict['EducationField'][EducationField]
    Gender=mapping_dict['Gender'][Gender]
    JobRole=mapping_dict['JobRole'][JobRole]
    MaritalStatus=mapping_dict['MaritalStatus'][MaritalStatus]
    OverTime=mapping_dict['OverTime'][OverTime]


    
    

    print(Age,BusinessTravel,DailyRate,Department,DistanceFromHome,Education,EducationField,
    EnvironmentSatisfaction,Gender,HourlyRate, JobInvolvement,JobRole,JobSatisfaction,MaritalStatus,
    MonthlyIncome,MonthlyRate,NumCompaniesWorked,OverTime,PercentSalaryHike,RelationshipSatisfaction,
    StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,
    YearsSinceLastPromotion,YearsWithCurrManager)

    output=model.predict([[Age,BusinessTravel,DailyRate,Department,DistanceFromHome,Education,EducationField,
    EnvironmentSatisfaction,Gender,HourlyRate, JobInvolvement,JobRole,JobSatisfaction,MaritalStatus,
    MonthlyIncome,MonthlyRate,NumCompaniesWorked,OverTime,PercentSalaryHike,RelationshipSatisfaction,
    StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,
    YearsSinceLastPromotion,YearsWithCurrManager]])


    if output[0]==0:
        data='Employee will not leave comapny.'
    else:
        data='Employee will leave'

    return render_template('pred.html',data=data)


if __name__  == "__main__":
    app.run(debug=True)