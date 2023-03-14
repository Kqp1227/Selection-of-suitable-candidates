% 导入数据：
% 采用"table"和"column"两种形式直接import
% "aug_train.csv"和"aug_test.csv"两个数据文件

%%
% target比例
tar1_per = sum(target(target==1))/length(target);
tar0_per = 1-tar1_per;

fprintf("target=1 情况的占比为:")
fprintf(num2str(tar1_per*100))
disp("%")

fprintf("target=0 情况的占比为:")
fprintf(num2str(tar0_per*100))
disp("%")

pie([tar1_per,tar0_per])
legend("target1","target2")
title("target比例分布柄图")

%%
% 数据清洗 - 去除缺省值
TF_augtrain = sum(ismissing(augtrain));
augtrain_without_missing = fillmissing(augtrain,'previous');

%%
% 需要手动替换"9 10 11"三列的首数据
TF_augtrain_without_missing = sum(ismissing(augtrain_without_missing))

%% 
% 去除enrollee_id数据
augtrain_without_missing_1 = augtrain_without_missing;
augtrain_without_missing_1(:,[1])=[];

%%
% 将categorical类型数据转换为用于区分的数值
gender_wm = table2array(augtrain_without_missing_1(:,[3]));
relevent_experience_wm = table2array(augtrain_without_missing_1(:,[4]));
enrolled_university_wm = table2array(augtrain_without_missing_1(:,[5]));
education_level_vm = table2array(augtrain_without_missing_1(:,[6]));
major_discipline_wm = table2array(augtrain_without_missing_1(:,[7]));
company_type_wm = table2array(augtrain_without_missing_1(:,[10]));

for i=1:19158
    if gender_wm(i)=="Male"
        gender_wm_d(i) = 1;
    else
        gender_wm_d(i) = 0;
    end
end
gender_wm_d = gender_wm_d';

for i=1:19158
    if relevent_experience_wm(i)=="Has relevent experience"
        relevent_experience_wm_d(i) = 1;
    else
        relevent_experience_wm_d(i) = 0;
    end
end
relevent_experience_wm_d = relevent_experience_wm_d';

for i=1:19158
    if enrolled_university_wm(i)=="Full time course"
        enrolled_university_wm_d(i) = 2;
    elseif enrolled_university_wm(i)=="Part time course"
        enrolled_university_wm_d(i) = 1;
    else
        enrolled_university_wm_d(i) = 0;
    end
end
enrolled_university_wm_d = enrolled_university_wm_d';

for i=1:19158
    if education_level_vm(i)=="Phd"
        education_level_vm_d(i) = 4;
    elseif education_level_vm(i)=="Masters"
        education_level_vm_d(i) = 3;
    elseif education_level_vm(i)=="Graduate"
        education_level_vm_d(i) = 2;
    elseif education_level_vm(i)=="High School"
        education_level_vm_d(i) = 1; 
    else
        education_level_vm_d(i) = 0; 
    end
end
education_level_vm_d = education_level_vm_d';

for i=1:19158
    if major_discipline_wm(i)=="STEM"
        major_discipline_wm_d(i) = 5;
    elseif major_discipline_wm(i)=="Arts"
        major_discipline_wm_d(i) = 4;
    elseif major_discipline_wm(i)=="Business Degree"
        major_discipline_wm_d(i) = 3;
    elseif major_discipline_wm(i)=="Humanities"
        major_discipline_wm_d(i) = 2;   
    elseif major_discipline_wm(i)=="Other"
        major_discipline_wm_d(i) = 1;   
    else
        major_discipline_wm_d(i) = 0;   
    end
end
major_discipline_wm_d = major_discipline_wm_d';

for i=1:19158
    if company_type_wm(i)=="Pvt Ltd"
        company_type_wm_d(i) = 5;
    elseif company_type_wm(i)=="Early Stage Startup"
        company_type_wm_d(i) = 4;
    elseif company_type_wm(i)=="Public Sector"
        company_type_wm_d(i) = 3;
    elseif company_type_wm(i)=="NGO"
        company_type_wm_d(i) = 2;
    elseif company_type_wm(i)=="Funded Startup"
        company_type_wm_d(i) = 1;
    else
        company_type_wm_d(i) = 0;
    end
end
company_type_wm_d = company_type_wm_d';

%%
city_wm = table2array(augtrain_without_missing_1(:,[1]));
city_development_index_wm = table2array(augtrain_without_missing_1(:,[2]));
experience_wm = table2array(augtrain_without_missing_1(:,[8]));
company_size_wm = table2array(augtrain_without_missing_1(:,[9]));
last_new_job_wm = table2array(augtrain_without_missing_1(:,[11]));
training_hours_wm = table2array(augtrain_without_missing_1(:,[12]));

%%
augtrain_without_missing_dd = [city_wm,city_development_index_wm,gender_wm_d,relevent_experience_wm_d,enrolled_university_wm_d,education_level_vm_d,major_discipline_wm_d,experience_wm,company_size_wm,company_type_wm_d,last_new_job_wm,training_hours_wm];

%%
for i=1:12
    r_all(i) = abs(corr(augtrain_without_missing_dd(:,[i]),target,'type','Spearman'));
end
r_all = r_all'

%%
% heatmap
yvalues = {'target'};
xvalues = {'city','city-development-index','gender','relevent-experience','enrolled-university','education-level','major-discipline','experience','company-size','company-type','last-new-job','training-hours'};
h = heatmap(xvalues,yvalues,r_all);


%%
% city
city_wm_1 = [];
city_wm_0 = [];
for i=1:19158
    if target(i)==1
        city_wm_1 = [city_wm_1,city_wm(i)];
    else
        city_wm_0 = [city_wm_0,city_wm(i)];
    end
end
hold on;
x = 0:180/14:180
a = hist(city_wm_1',15);
plot(x,a)
b = hist(city_wm_0',15);
plot(x,b)
legend("target=1","target=0")
title("density person change a job based on city")

%%
% city development index
city_development_index_1 = [];
city_development_index_0 = [];
for i=1:19158
    if target(i)==1
        city_development_index_1 = [city_development_index_1,city_development_index_wm(i)];
    else
        city_development_index_0 = [city_development_index_0,city_development_index_wm(i)];
    end
end
hold on;
x = 0:1/14:1
a = hist(city_development_index_1',15);
plot(x,a)
b = hist(city_development_index_0',15);
plot(x,b)
legend("target=1","target=0")
title("density person change a job based on city development index")

%%
% gender
gender_1 = [];
gender_0 = [];
for i=1:19158
    if target(i)==1
        gender_1 = [gender_1,gender_wm(i)];
    else
        gender_0 = [gender_0,gender_wm(i)];
    end
end
Y1 = hist(gender_1);
Y2 = hist(gender_0);
bar([Y1;Y2]');
set(gca,'XTickLabel',{'Female','Male','Other'})
legend("target=1","target=0")
title("density person change a job based on gender")

%%
% relevent_experience
relevent_experience_1 = [];
relevent_experience_0 = [];
for i=1:19158
    if target(i)==1
        relevent_experience_1 = [relevent_experience_1,relevent_experience_wm(i)];
    else
        relevent_experience_0 = [relevent_experience_0,relevent_experience_wm(i)];
    end
end
relevent_experience_Y1 = hist(relevent_experience_1);
relevent_experience_Y2 = hist(relevent_experience_0);
bar([relevent_experience_Y1;relevent_experience_Y2]');
set(gca,'XTickLabel',{'Has relevent experience','No relevent experience'})
legend("target=1","target=0")
title("density person change a job based on relevent experience")

%%
% enrolled university
enrolled_university_1 = [];
enrolled_university_0 = [];
for i=1:19158
    if target(i)==1
        enrolled_university_1 = [enrolled_university_1,enrolled_university_wm(i)];
    else
        enrolled_university_0 = [enrolled_university_0,enrolled_university_wm(i)];
    end
end
enrolled_university_Y1 = hist(enrolled_university_1);
enrolled_university_Y2 = hist(enrolled_university_0);
bar([enrolled_university_Y1;enrolled_university_Y2]');
set(gca,'XTickLabel',{'Full time course','Part time course','no enrollment'})
legend("target=1","target=0")
title("density person change a job based on enrolled university")

%%
% education level
education_level_1 = [];
education_level_0 = [];
for i=1:19158
    if target(i)==1
        education_level_1 = [education_level_1,education_level_vm(i)];
    else
        education_level_0 = [education_level_0,education_level_vm(i)];
    end
end
education_level_Y1 = hist(education_level_1);
education_level_Y2 = hist(education_level_0);
bar([education_level_Y1;education_level_Y2]');
set(gca,'XTickLabel',{'Graduate','High school','Masters','Phd','Primary School'})
legend("target=1","target=0")
title("density person change a job based on education level")

%%
% major discipline
major_discipline_1 = [];
major_discipline_0 = [];
for i=1:19158
    if target(i)==1
        major_discipline_1 = [major_discipline_1,major_discipline_wm(i)];
    else
        major_discipline_0 = [major_discipline_0,major_discipline_wm(i)];
    end
end
major_discipline_Y1 = hist(major_discipline_1);
major_discipline_Y2 = hist(major_discipline_0);
bar([major_discipline_Y1;major_discipline_Y2]');
set(gca,'XTickLabel',{'No Major','Bussiness Degree','Humanities','Arts','Other','STEM'})
legend("target=1","target=0")
title("density person change a job based on major discipline")

%%
% experience
experience_1 = [];
experience_0 = [];
for i=1:19158
    if target(i)==1
        experience_1 = [experience_1,experience_wm(i)];
    else
        experience_0 = [experience_0,experience_wm(i)];
    end
end
hold on;
x = 0:24/8:24
a = hist(experience_1',9);
plot(x,a)
b = hist(experience_0',9);
plot(x,b)
legend("target=1","target=0")
title("density person change a job based on city experience")

%%
% company size
company_size_1 = [];
company_size_0 = [];
for i=1:19158
    if target(i)==1
        company_size_1 = [company_size_1,company_size_wm(i)];
    else
        company_size_0 = [company_size_0,company_size_wm(i)];
    end
end
hold on;
x = 0:10000/10:10000
a = hist(company_size_1',11);
plot(x,a)
b = hist(company_size_0',11);
plot(x,b)
legend("target=1","target=0")
title("density person change a job based on company size")

%%
% company type
company_type_1 = [];
company_type_0 = [];
for i=1:19158
    if target(i)==1
        company_type_1 = [company_type_1,company_type_wm(i)];
    else
        company_type_0 = [company_type_0,company_type_wm(i)];
    end
end
company_type_Y1 = hist(company_type_1);
company_type_Y2 = hist(company_type_0);
bar([company_type_Y1;company_type_Y2]');
set(gca,'XTickLabel',{'NGO','Funded Startup','Early Stage Startup','Other','Public Sector','Pvt Ltd'})
legend("target=1","target=0")
title("density person change a job based on company type")

%%
% last new job
last_new_job_1 = [];
last_new_job_0 = [];
for i=1:19158
    if target(i)==1
        last_new_job_1 = [last_new_job_1,last_new_job_wm(i)];
    else
        last_new_job_0 = [last_new_job_0,last_new_job_wm(i)];
    end
end
last_new_job_Y1 = hist(last_new_job_1);
last_new_job_Y2 = hist(last_new_job_0);
bar([last_new_job_Y1;last_new_job_Y2]');
set(gca,'XTickLabel',{'1','','','2','','','3','','','4','',''})
legend("target=1","target=0")
title("density person change a job based on last new job")

%%
% training hours
training_hours_1 = [];
training_hours_0 = [];
for i=1:19158
    if target(i)==1
        training_hours_1 = [training_hours_1,training_hours_wm(i)];
    else
        training_hours_0 = [training_hours_0,training_hours_wm(i)];
    end
end
hold on;
x = 0:350/10:350
a = hist(training_hours_1',11);
plot(x,a)
b = hist(training_hours_0',11);
plot(x,b)
legend("target=1","target=0")
title("density person change a job based on training hours")

%%
augtrain_without_missing_ddd = [augtrain_without_missing_dd,target];

%%
% double转换为table类型
aug = array2table(augtrain_without_missing_ddd,'VariableNames',{'city','city_development_index','gender','relevent_experience','enrolled_university','education_level','major_discipline','experience','company_size','company_type','last_new_job','training_hours','target'});

%%
% SVM
inputTable = aug;
predictorNames = {'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours'};
predictors = inputTable(:, predictorNames);
response = inputTable.target;
isCategoricalPredictor = [false, false, true, true, true, true, true, false, false, true, false, false];

classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);

predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

trainedClassifier.RequiredVariables = {'city', 'city_development_index', 'company_size', 'company_type', 'education_level', 'enrolled_university', 'experience', 'gender', 'last_new_job', 'major_discipline', 'relevent_experience', 'training_hours'};
trainedClassifier.ClassificationSVM = classificationSVM;

inputTable = aug;
predictorNames = {'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours'};
predictors = inputTable(:, predictorNames);
response = inputTable.target;
isCategoricalPredictor = [false, false, true, true, true, true, true, false, false, true, false, false];

% cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 4)

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')

%%
% Linear KNN
inputTable = aug;
predictorNames = {'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours'};
predictors = inputTable(:, predictorNames);
response = inputTable.target;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];

classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);

predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.RequiredVariables = {'city', 'city_development_index', 'company_size', 'company_type', 'education_level', 'enrolled_university', 'experience', 'gender', 'last_new_job', 'major_discipline', 'relevent_experience', 'training_hours'};
trainedClassifier.ClassificationKNN = classificationKNN;

inputTable = aug;
predictorNames = {'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours'};
predictors = inputTable(:, predictorNames);
response = inputTable.target;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 5)

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')

%%
% Medium KNN
inputTable = aug;
predictorNames = {'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours'};
predictors = inputTable(:, predictorNames);
response = inputTable.target;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];

classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);

predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.RequiredVariables = {'city', 'city_development_index', 'company_size', 'company_type', 'education_level', 'enrolled_university', 'experience', 'gender', 'last_new_job', 'major_discipline', 'relevent_experience', 'training_hours'};
trainedClassifier.ClassificationKNN = classificationKNN;

inputTable = aug;
predictorNames = {'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours'};
predictors = inputTable(:, predictorNames);
response = inputTable.target;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 5)

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')

%%
TF_augtest = sum(ismissing(augtest));
augtest_without_missing = fillmissing(augtest,'previous');

TF_augtest_without_missing = sum(ismissing(augtest_without_missing))

%%
% 去除enrollee_id数据
augtest_without_missing_1 = augtest_without_missing;
augtest_without_missing_1(:,[1])=[];

%%
% 将categorical类型数据转换为用于区分的数值
test_gender_wm = table2array(augtest_without_missing_1(:,[3]));
test_relevent_experience_wm = table2array(augtest_without_missing_1(:,[4]));
test_enrolled_university_wm = table2array(augtest_without_missing_1(:,[5]));
test_education_level_vm = table2array(augtest_without_missing_1(:,[6]));
test_major_discipline_wm = table2array(augtest_without_missing_1(:,[7]));
test_company_type_wm = table2array(augtest_without_missing_1(:,[10]));

%%
for i=1:2129
    if test_gender_wm(i)=="Male"
        test_gender_wm_d(i) = 1;
    else
        test_gender_wm_d(i) = 0;
    end
end
test_gender_wm_d = test_gender_wm_d';

for i=1:2129
    if test_relevent_experience_wm(i)=="Has relevent experience"
        test_relevent_experience_wm_d(i) = 1;
    else
        test_relevent_experience_wm_d(i) = 0;
    end
end
test_relevent_experience_wm_d = test_relevent_experience_wm_d';

for i=1:2129
    if test_enrolled_university_wm(i)=="Full time course"
        test_enrolled_university_wm_d(i) = 2;
    elseif test_enrolled_university_wm(i)=="Part time course"
        test_enrolled_university_wm_d(i) = 1;
    else
        test_enrolled_university_wm_d(i) = 0;
    end
end
test_enrolled_university_wm_d = test_enrolled_university_wm_d';

for i=1:2129
    if test_education_level_vm(i)=="Phd"
        test_education_level_vm_d(i) = 4;
    elseif test_education_level_vm(i)=="Masters"
        test_education_level_vm_d(i) = 3;
    elseif test_education_level_vm(i)=="Graduate"
        test_education_level_vm_d(i) = 2;
    elseif test_education_level_vm(i)=="High School"
        test_education_level_vm_d(i) = 1; 
    else
        test_education_level_vm_d(i) = 0; 
    end
end
test_education_level_vm_d = test_education_level_vm_d';

for i=1:2129
    if test_major_discipline_wm(i)=="STEM"
        test_major_discipline_wm_d(i) = 5;
    elseif test_major_discipline_wm(i)=="Arts"
        test_major_discipline_wm_d(i) = 4;
    elseif test_major_discipline_wm(i)=="Business Degree"
        test_major_discipline_wm_d(i) = 3;
    elseif test_major_discipline_wm(i)=="Humanities"
        test_major_discipline_wm_d(i) = 2;   
    elseif test_major_discipline_wm(i)=="Other"
        test_major_discipline_wm_d(i) = 1;   
    else
        test_major_discipline_wm_d(i) = 0;   
    end
end
test_major_discipline_wm_d = test_major_discipline_wm_d';

for i=1:2129
    if test_company_type_wm(i)=="Pvt Ltd"
        test_company_type_wm_d(i) = 5;
    elseif test_company_type_wm(i)=="Early Stage Startup"
        test_company_type_wm_d(i) = 4;
    elseif test_company_type_wm(i)=="Public Sector"
        test_company_type_wm_d(i) = 3;
    elseif test_company_type_wm(i)=="NGO"
        test_company_type_wm_d(i) = 2;
    elseif test_company_type_wm(i)=="Funded Startup"
        test_company_type_wm_d(i) = 1;
    else
        test_company_type_wm_d(i) = 0;
    end
end
test_company_type_wm_d = test_company_type_wm_d';

%%
test_city_wm = table2array(augtest_without_missing_1(:,[1]));
test_city_development_index_wm = table2array(augtest_without_missing_1(:,[2]));
test_experience_wm = table2array(augtest_without_missing_1(:,[8]));
test_company_size_wm = table2array(augtest_without_missing_1(:,[9]));
test_last_new_job_wm = table2array(augtest_without_missing_1(:,[11]));
test_training_hours_wm = table2array(augtest_without_missing_1(:,[12]));

%%
augtest_without_missing_dd = [test_city_wm,test_city_development_index_wm,test_gender_wm_d,test_relevent_experience_wm_d,test_enrolled_university_wm_d,test_education_level_vm_d,test_major_discipline_wm_d,test_experience_wm,test_company_size_wm,test_company_type_wm_d,test_last_new_job_wm,test_training_hours_wm];

%%
% 调用训练好的 Medium KNN模型
y_target = predict(classificationKNN,augtest_without_missing_dd)

% 写入csv文件
y_target_table = array2table(y_target,'VariableNames',{'predicted_y_target'});
aug_test_con = [augtest, y_target_table];
writetable(aug_test_con, "aug_test_con.csv")