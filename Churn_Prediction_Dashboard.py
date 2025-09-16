import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

st.title('Churn Prediction Dashboard')

# Get the dataset
telco_data = pd.read_csv('telco_data.csv')

############## Cleaning data #####################
# Clean and convert TotalCharges column
tc = telco_data['TotalCharges'].str.strip()
tc = telco_data['TotalCharges'] = pd.to_numeric(telco_data['TotalCharges'], errors="coerce")
telco_data['TotalCharges'] = tc

telco_data["TotalCharges"] = telco_data['TotalCharges'].fillna(value=telco_data['TotalCharges'].mean())

# Create a binary churn flag for Churn column
telco_data['ChurnFlag'] = telco_data['Churn'].map({'Yes': 1, 'No': 0})

############ Target Overview #####################
#Custom color map
color_map = {
    'Yes': "#442020",
    'No': "#314A31"
}

# Add Churn Rate Pie
pie = px.pie(
    data_frame=telco_data,
    names='Churn',
    title='Churn Rate Review',
    color='Churn',
    color_discrete_map=color_map
)

st.plotly_chart(pie)

selected_anaysis_type = st.selectbox(
    options=[
        'Choose Analysis Type', 
        'Numeric Drivers of Churn', 
        'Categorical Drivers of Churn',
        'Service Engagement and Churn',
        'Demographic Drivers of Churn',
        'Correlations and Interactions'],
    label='Choose Analysis Type',
    label_visibility='collapsed',
    index=0
)

model_name = st.selectbox(
    options=[
        'Choose Model',
        'Logistic Regression',
        'Random Forest'
    ],
    label='Model',
    index=0
)

if selected_anaysis_type == 'Numeric Drivers of Churn':

    ############## Key numeric features ################
    st.subheader(selected_anaysis_type)

    tenure_box_col, tenure_note_col = st.columns([5, 2])

    # Custom color map
    color_map = {
        'Yes': "#8C4141",
        'No': "#507B50"
    }

    # Add boxplot for key numeric 1
    tenure_box = px.box(
        data_frame=telco_data,
        x=telco_data['Churn'],
        y=telco_data['tenure'],
        color='Churn',
        color_discrete_map=color_map,
        title='Tenure Distribution by Churn Status'
    )

    tenure_box.update_layout(yaxis_title='Tenure (Months)')

    tenure_box_col.plotly_chart(tenure_box)

    tenure_note_col.text('Customers who stayed longer with the company are less likely to churn. ' \
    'The boxplot shows that churners typically had shorter tenures compared to those who remain.')

    # Add histogram for key numeric 2
    monthly_charges_col, monthly_charges_note_col = st.columns([5, 2])

    monthly_charges_hist = px.histogram(
        data_frame=telco_data,
        x='MonthlyCharges',
        color='Churn',
        color_discrete_map=color_map,
        nbins=50,
        histnorm="percent",
        title='Monthly Charges Distribution by Churn Status',
        labels={
            "MonthlyCharges": "Monthly Charges ($)"
        }
    )

    monthly_charges_hist.update_layout(yaxis_title='Percentage of Customers')

    monthly_charges_col.plotly_chart(monthly_charges_hist)

    monthly_charges_note_col.text('Higher monthly charges are associated with a greater likelihood of churn. ' \
    'Churned customers are more concentrated in the higher charge ranges, while non-churners are more evenly ' \
    'distributed across the lower ranges.')

    # Add scatter for key numeric 3

    total_charges_tenure_col, total_charges_tenure_note_col = st.columns([5, 2])

    total_charges_tenure_scatter = px.scatter(
        data_frame=telco_data,
        x='TotalCharges',
        y='tenure',
        color='Churn',
        color_discrete_map=color_map,
        title='Relationship Between Tenure and Total Charges by Churn Status',
        labels={
            'TotalCharges': 'Total Charges ($)',
            'tenure': 'Tenure (Months)'
            }
    )

    total_charges_tenure_col.plotly_chart(total_charges_tenure_scatter)

    total_charges_tenure_note_col.text('There is a strong positive relationship between tenure and total charges, ' \
    'as expected. However, churners cluster more heavily in the lower-tenure, lower-total-charge region, indicating ' \
    'that newer customers with less accumulated spend are more prone to churn.')

elif selected_anaysis_type == 'Categorical Drivers of Churn':
    ###################### High-impact categoricals ####################
    st.subheader(selected_anaysis_type)

    categorical1, categorical2 = st.columns([1, 1])

    # Add contract vs churn bar
    contract_bar = px.bar(
        data_frame=telco_data,
        x='Contract',
        color='Churn',
        color_discrete_map=color_map,
        title='Churn Rates by Contract Type'
    )

    contract_bar.update_layout(yaxis_title='Number of Customers')

    categorical1.plotly_chart(contract_bar)

    categorical1.text('Customers on month-to-month contracts are far more likely to churn than those ' \
    'on one-year or two-year contracts. Longer contracts appear to reduce churn, likely because they lock '
    'in commitment and possibly offer discounts.')

    # Add payment method vs churn bar
    payment_method_bar = px.bar(
        data_frame=telco_data,
        x='PaymentMethod',
        color='Churn',
        color_discrete_map=color_map,
        title='Churn Rates by Payment Method'
    )

    contract_bar.update_layout(yaxis_title='Number of Customers')

    categorical2.plotly_chart(payment_method_bar)

    categorical2.text('Churn is highest among customers using electronic checks, while those paying by ' \
    'credit card or bank transfer (automatic) churn less. This may reflect both customer demographics and ' \
    'the convenience/reliability of automated payments.')



    categorical3, categorical4 = st.columns([1, 1])

    # Add paperless billing vs churn bar
    paperless_billing_bar = px.bar(
        data_frame=telco_data,
        x='PaperlessBilling',
        color='Churn',
        color_discrete_map=color_map,
        title='Churn Rates by Paperless Billing',
        barmode='group'
    )

    paperless_billing_bar.update_layout(yaxis_title='Number of Customers')

    categorical3.plotly_chart(paperless_billing_bar)

    categorical3.text('Customers with paperless billing have noticeably higher churn than those receiving paper bills. ' \
    'This might indicate differences in customer profiles: paperless billing users could be younger, more price-sensitive, '
    'or more likely to switch providers.')

    # Add internet service vs churn bar
    internet_service_bar = px.bar(
        data_frame=telco_data,
        x='InternetService',
        color='Churn',
        color_discrete_map=color_map,
        title='Churn Rates by Internet Service',
        barmode='group'
    )

    internet_service_bar.update_layout(yaxis_title='Number of Customers')

    categorical4.plotly_chart(internet_service_bar)

    categorical4.text('Churn is most pronounced among Fiber optic users, while DSL customers show lower churn, '
    'and those without internet service churn the least. This suggests that fiber customers may have higher' \
    ' expectations for service or face more competitive alternatives.')

elif selected_anaysis_type == 'Service Engagement and Churn':
    st.subheader(selected_anaysis_type)

    services_count_line_col, services_count_note_col = st.columns([5, 2])

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines"
    ]
    
    # Create a new column with the count of "Yes"
    telco_data["ServicesCount"] = telco_data[service_cols].apply(lambda row: (row == "Yes").sum(), axis=1)

    # Group by ServicesCount and calculate churn rate
    churn_rate_by_services = (
        telco_data.groupby("ServicesCount")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)  # percentage
        .reset_index(name="ChurnRate")
)
    
    services_count_line = px.line(
        churn_rate_by_services,
        x="ServicesCount",
        y="ChurnRate",
        markers=True,
        title="Churn Rate by Number of Services",
        labels={
            "ServicesCount": "Number of Services Subscribed",
            "ChurnRate": "Churn Rate (%)"
        }
)

    services_count_line_col.plotly_chart(services_count_line)

    services_count_note_col.text('Customers with fewer subscribed services are more likely to churn, ' \
    'with churn rates peaking among those with only one or two services. As the number of services ' \
    'increases, the churn rate steadily declines, suggesting that broader service engagement creates ' \
    'stronger customer loyalty and reduces the likelihood of leaving.')

elif selected_anaysis_type == 'Demographic Drivers of Churn':
    st.subheader(selected_anaysis_type)

    demographic_drivers_bar_col, demographic_drivers_note_col = st.columns([5, 2])

    churned_customers = telco_data[telco_data['Churn'] == 'Yes']

    telco_data['SeniorCitizenFlag'] = telco_data['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    
    demo_cols = ["SeniorCitizenFlag", "Partner", "Dependents"]  # adjust as needed

    # Build churn rate table for each feature and its categories
    color_map = {
    'Yes': "#a26c5c",  # orange
    "No": "#69988C"            # green
    }
    
    #
    frames = []
    for col in demo_cols:
        tmp = (
            telco_data
            .groupby(col)["Churn"]
            .apply(lambda s: (s == "Yes").mean()*100)
            .reset_index()
            .rename(columns={col: "Category", "Churn": "ChurnRate"})
        )
        tmp["Feature"] = col
        frames.append(tmp)

    rates = pd.concat(frames, ignore_index=True)

    demographic_drivers_bar = px.bar(
        rates,
        x="Feature",
        y="ChurnRate",
        color="Category",
        color_discrete_map=color_map,
        barmode="group",
        text=rates["ChurnRate"].round(1).astype(str) + "%",
        title="Who Churns More (Churn Rate by Demographic Group)",
        labels={"Feature": "Demographic Feature", "ChurnRate": "Churn Rate (%)"}
    )

    demographic_drivers_bar.update_traces(textposition="outside")
    demographic_drivers_bar.update_layout(yaxis_range=[0, max(5, rates["ChurnRate"].max()*1.15)])

    demographic_drivers_bar_col.plotly_chart(demographic_drivers_bar)

    demographic_drivers_note_col.text('Demographic factors play an important role in churn behavior. ' \
    'Senior citizens show the highest churn rate (41.7%), indicating they are more likely to leave. ' \
    'In contrast, customers with a partner (19.7%) or dependents (15.5%) are less likely to churn ' \
    'compared to those without. This suggests that customers with family responsibilities tend to be ' \
    'more stable and loyal, while older customers may be more prone to switching providers.')

elif selected_anaysis_type == 'Correlations and Interactions':
    st.subheader('Correlations and Interactions')

    corr_column1, corr_column2 = st.columns([1, 1])

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag"]

    # Compute correlation matrix
    correlation_matrix = telco_data[numeric_cols].corr(numeric_only=True)

    # Plot heatmap
    heatmap = px.imshow(
        correlation_matrix.round(2),
        text_auto=True,                  # show correlation values
        color_continuous_scale="icefire", # red = negative, blue = positive
        title="Correlation Heatmap of Numeric Features"
    )

    heatmap.update_layout(
        xaxis_title="Features",
        yaxis_title="Features"
    )

    corr_column1.plotly_chart(heatmap)

    corr_column1.text('The heatmap shows strong positive correlations between tenure and total charges (0.82), ' \
    'as expected, since customers who stay longer accumulate higher charges. Monthly charges are moderately ' \
    'correlated with total charges (0.65) but less related to tenure (0.25). Importantly, churn flag is negatively ' \
    'correlated with tenure (-0.35), meaning longer-tenure customers are less likely to churn, while its correlation ' \
    'with monthly charges (0.19) suggests higher monthly fees slightly increase churn risk.')

    # 1) Define tenure buckets
    bins = [0, 6, 12, 24, 48, 72]              # adjust if your max tenure differs
    labels = ["0–6", "7–12", "13–24", "25–48", "49–72"]

    telco_data["TenureBucket"] = pd.cut(
        telco_data["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    # 2) Compute churn rate per bucket
    # Assumes Churn values are "Yes"/"No"
    bucket_rates = (
        telco_data.groupby("TenureBucket")["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )

    # Ensure buckets are in the intended order
    bucket_rates["TenureBucket"] = pd.Categorical(bucket_rates["TenureBucket"], categories=labels, ordered=True)
    bucket_rates = bucket_rates.sort_values("TenureBucket")

    # 3a) Bar chart (presentation-friendly)
    fig_bar = px.bar(
        bucket_rates,
        x="TenureBucket",
        y="ChurnRate",
        text=bucket_rates["ChurnRate"].round(1).astype(str) + "%",
        title="Churn Rate by Tenure Bucket",
        labels={"TenureBucket": "Tenure (months)", "ChurnRate": "Churn Rate (%)"},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(yaxis_range=[0, max(5, bucket_rates["ChurnRate"].max() * 1.15)])
    fig_bar.show()

    # 3b) Line chart (to show shape clearly) — optional
    fig_line = px.line(
        bucket_rates,
        x="TenureBucket",
        y="ChurnRate",
        markers=True,
        title="Churn Rate by Tenure Bucket (Trend)",
        labels={"TenureBucket": "Tenure (months)", "ChurnRate": "Churn Rate (%)"},
    )
    fig_line.update_layout(yaxis_range=[0, max(5, bucket_rates["ChurnRate"].max() * 1.15)])

    corr_column2.plotly_chart(fig_line)

    corr_column2.text('Churn risk decreases sharply as customer tenure increases. Customers in their first ' \
    '6 months churn at over 50%, while those with 1–2 years of tenure churn around 30%. After 2 years, ' \
    'churn steadily declines, reaching under 10% for customers with more than 4 years of tenure. This highlights ' \
    'the importance of retention strategies in the early stages of the customer lifecycle, where churn risk is highest.')

######################## Modeling ####################################################################

####################### Encoding data ###############################################
# Encode binary data
mapping = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}

cols_to_map = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MultipleLines'
]

for col in cols_to_map:
    if col == 'gender':
        telco_data[col] = telco_data[col].map({'Female': 1, 'Male': 0})
    else:
        telco_data[col] = telco_data[col].map(mapping)

# Encode multi-category columns
telco_data_encoded = pd.get_dummies(
    data=telco_data,
    columns=["Contract", "PaymentMethod", "InternetService"],
    drop_first=True
)

# force dummies to int, leave other columns untouched
dummy_cols = telco_data_encoded.columns.difference(telco_data.columns)
telco_data_encoded[dummy_cols] = telco_data_encoded[dummy_cols].astype(int)

# Drop customer id column
telco_data_encoded = telco_data_encoded.drop(['customerID', 'Churn'], axis=1)

# Scaling numerical data
scaler = StandardScaler()

# Features (exclude target column)
features = telco_data_encoded.drop('ChurnFlag', axis=1)
numeric_cols = features.select_dtypes(include=[np.number]).columns

# Fit and transform
scaled_features = scaler.fit_transform(features[numeric_cols])

# Build DataFrame with correct column names
telco_data_features = pd.DataFrame(scaled_features, columns=numeric_cols)

# Split the data
X = telco_data_features
y = telco_data_encoded['ChurnFlag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#################### Training and evaluation ########################################3

# Logistic Regression
if model_name == 'Logistic Regression':
    # Train
    log_reg_model = LogisticRegression()

    log_reg_model.fit(X_train, y_train)

    log_reg_predictions = log_reg_model.predict(X_test)

    st.subheader('Logistic Regression Model Evaluation')
    log_reg_evaluation_col, lo_reg_comment_col = st.columns([5, 3])

    # Evaluate
    log_reg_report = classification_report(
        y_test, 
        log_reg_predictions, 
        digits=3, 
        target_names=["No churn","Churn"],
        output_dict=True
        )
    
    df_report = pd.DataFrame(log_reg_report).T

    log_reg_evaluation_col.dataframe(df_report, use_container_width=True)
    lo_reg_comment_col.text('The model achieves 82.0% accuracy on an imbalanced set (1,036 non-churn vs 373 churn). ' \
    'It identifies non-churners reliably (precision 86.1%, recall 90.2%). For churners, precision is 68.5% and recall '
    'is 59.5% (F1 63.7%), meaning ~4/10 churners are missed and ~1/3 churn flags are false alarms at this cutoff.')

    # ROC-AUC
    
    # Get probability that that customer will churn
    positive_class_index  = list(log_reg_model.classes_).index(1)   # robust: find where class "1" is
    churn_probability = log_reg_model.predict_proba(X_test)[:, positive_class_index]

    # AUC number

    # Check how well the predicted probabilities line up with the actual churn labels
    auc = roc_auc_score(y_test, churn_probability)

    # ROC curve with Plotly

    # x/y coordinates to plot the ROC curve.
    # They’re computed from true labels (y_test) and predicted probabilities (churn_probability)
    false_positive, true_positive, _ = roc_curve(y_test, churn_probability)

    roc_line = px.line(
        pd.DataFrame({"FPR": false_positive, "TPR": true_positive}),
        x="FPR", y="TPR",
        title=f"ROC Curve — Logistic Regression (AUC = {auc:.3f})",
        labels={"FPR":"False Positive Rate", "TPR":"True Positive Rate"}
    )
    roc_line.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    roc_line.update_layout(xaxis_range=[0,1], yaxis_range=[0,1])

    st.plotly_chart(roc_line, use_container_width=True)
    st.text('The model shows strong ranking performance with AUC = 0.862, meaning ' \
    'it can reliably score churners above non-churners. The blue curve sits well above the dashed ' \
    'random baseline. Around FPR ≈ 10%, the TPR is ~60–65%; at FPR ≈ 20%, TPR exceeds ~75%, indicating ' \
    'you can capture many churners with relatively few false alarms. The “knee” of the curve is roughly '
    'in the 0.10–0.20 FPR range— a sensible region to choose an operating threshold depending on your tolerance for false positives.')

    # Explainability
    st.subheader('Explainability')
    tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])

    # Small background sample for SHAP (keeps it fast & stable)
    background_features = X_train.sample(min(1000, len(X_train)), random_state=42)

    explainer = shap.Explainer(log_reg_model, background_features )

    tab_global, tab_local = tab_shap.tabs(["Global importance", "Per-customer"])

    # ---------- GLOBAL IMPORTANCE ----------
    tab_global.subheader('Global Feature Importance — Logistic Regression Model (SHAP)')

    # explain a small evaluation sample (speed)
    features_for_global  = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_global_explanations  = explainer(features_for_global)  # shap.Explanation

    mean_abs_contrib_by_feature = np.abs(shap_global_explanations.values).mean(axis=0)

    importance_table = (
        pd.DataFrame({
            "feature": features_for_global.columns,
            "avg_abs_contribution": mean_abs_contrib_by_feature
        })
        .sort_values("avg_abs_contribution", ascending=False)
    )

    global_importance_bar = px.bar(
        importance_table.head(15).iloc[::-1],  # reverse so the biggest bar is on top
        x="avg_abs_contribution",
        y="feature",
        orientation="h",
        title=f"Top 15 Features by Global Importance (SHAP) — Logistic Regression Model",
        labels={"avg_abs_contribution": "Average |contribution|", "feature": "Feature"}
        )  
    
    tab_global.plotly_chart(global_importance_bar, use_container_width=True)
    tab_global.caption("Average absolute SHAP value = how much a feature moves predictions on average. Bigger bar → more influence overall.")

    # ------------------- SHAP per customer --------------------------
    row_explanation = explainer(X_test.iloc[[0]])

    per_feature_contrib = (
        pd.DataFrame({
            "feature": X_test.columns,
            "shap_value": row_explanation.values[0],       # signed effect per feature
            "feature_value": X_test.iloc[[0]].iloc[0].values # the (scaled/encoded) input values
        })
        .assign(abs_effect=lambda d: d["shap_value"].abs())
        .sort_values("abs_effect", ascending=False)
        .drop(columns="abs_effect")
    )

    fig_shap_customer = px.bar(
        per_feature_contrib.head(12).iloc[::-1],
        x="shap_value",          # was shap_contribution
        y="feature",             # was feature_name
        orientation="h",
        title=f"Top 12 feature contributions — row 0",
        labels={"shap_value": "Contribution (→ churn + / ← non-churn −)", "feature": "Feature"}
    )

    tab_local.plotly_chart(fig_shap_customer, use_container_width=True)
    tab_local.text('Each bar shows how a feature pushed this customer’s prediction: right = ' \
    'toward churn, left = toward non-churn. The biggest push toward churn comes from tenure '
    '(likely short tenure) and MonthlyCharges (relatively high). Smaller pushes come from factors like ' \
    'PhoneService and Electronic check. The main factors reducing churn risk for this customer are ' \
    'InternetService_Fiber optic and TotalCharges (their values pull the prediction left). Magnitude = ' \
    'strength of influence; the signed contributions sum (with the baseline) to this customer’s final churn probability.')

    tab_local.dataframe(per_feature_contrib.head(12), use_container_width=True)
    tab_local.text('Each row shows how a feature moved this customer’s prediction: positive = pushes ' \
    'toward churn, negative = pushes toward non-churn. The biggest pushes toward churn are tenure (+1.68) '
    'and MonthlyCharges (+0.85), with smaller pushes from PhoneService, Contract_Two year, PaymentMethod_Electronic ' \
    'check, InternetService_No, and Contract_One year. The main factors reducing churn risk are InternetService_Fiber ' \
    'optic (–0.67), TotalCharges (–0.67), plus smaller negatives from MultipleLines, StreamingMovies, and StreamingTV. ' \
    'The last number is the standardized feature value for this customer (negative ≈ below the dataset average).')

    #----------------------- LIME -----------------------
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
         feature_names=X_train.columns.tolist(),
         class_names=["No churn","Churn"],
         discretize_continuous=True,
         mode="classification"
    )

    lime_explanation = lime_explainer.explain_instance(
        data_row=X_test.iloc[[0]].iloc[0].values,
        predict_fn=log_reg_model.predict_proba,
        num_features=10
     )
    
    lime_items = lime_explanation.as_list()  # list of (feature_or_rule, weight)
    lime_table = pd.DataFrame(lime_items, columns=["feature_or_rule", "lime_weight"])

    fig_lime_customer = px.bar(
        lime_table.iloc[::-1],
        x="lime_weight", y="feature_or_rule",
        orientation="h",
        title=f"LIME local explanation — row 0",
        labels={"lime_weight":"Local weight (→ churn + / ← non-churn −)", "feature_or_rule":"Feature / Rule"}
    )

    tab_lime.plotly_chart(fig_lime_customer, use_container_width=True)
    tab_lime.text('Each bar shows a simple rule near this customer and its local weight: right = pushes prediction toward ' \
    'churn, left = pushes away. Here, rules like tenure <= -0.95, MonthlyCharges <= -0.96, Contract_Two year <= -0.56, '
    'and InternetService_No <= -0.53 increase churn risk for this customer, while TotalCharges <= -0.83, InternetService_Fiber ' \
    'optic <= -0.89, and several others reduce it.')
    tab_lime.dataframe(lime_table, use_container_width=True)
    tab_lime.text('Each rule shows a simple condition near this customer and its local weight: positive = pushes prediction' \
    ' toward churn, negative = pushes toward non-churn. \nBiggest risk drivers here: very short tenure (tenure ≤ −0.95), ' \
    'below-average monthly charges (MonthlyCharges ≤ −0.96), not on a 2-year contract (Contract_Two year ≤ −0.56), and having ' \
    'internet service (InternetService_No ≤ −0.53 ⇒ not “No internet”).\nRisk-reducing factors: lower total charges (TotalCharges ≤ −0.83),' \
    ' not fiber-optic (InternetService_Fiber optic ≤ −0.89), plus small reductions from StreamingMovies and MultipleLines.')

# Random Forest
elif model_name == 'Random Forest':
    # Train
    random_forest_model = RandomForestClassifier(n_estimators=200)

    random_forest_model.fit(X_train, y_train)

    random_forest_predictions = random_forest_model.predict(X_test)

    st.subheader('Random Forest Model Evaluation')

    print(classification_report(y_test, random_forest_predictions))
    rand_forest_evaluation_col, rand_forest_comment_col = st.columns([5, 3])
    
    # Evaluate
    rand_forest_report = classification_report(
        y_test, 
        random_forest_predictions, 
        digits=3, 
        target_names=["No churn","Churn"],
        output_dict=True
        )
    
    df_report = pd.DataFrame(rand_forest_report).T

    rand_forest_evaluation_col.dataframe(df_report, use_container_width=True)
    rand_forest_comment_col.text('The model reaches 79.4% accuracy on an imbalanced set (1,036 non-churn vs 373 churn). ' \
    'It identifies non-churners well (precision 82.5%, recall 91.3%). For churners, precision is 65.8% and recall is ' \
    '46.4% (F1 54.4%), meaning about 53.6% of churners are missed at this cutoff.')

    # ROC-AUC
    
    # Get probability that that customer will churn
    positive_class_index  = list(random_forest_model.classes_).index(1)   # robust: find where class "1" is
    churn_probability = random_forest_model.predict_proba(X_test)[:, positive_class_index]

    # AUC number

    # Check how well the predicted probabilities line up with the actual churn labels
    auc = roc_auc_score(y_test, churn_probability)

    # ROC curve with Plotly

    # x/y coordinates to plot the ROC curve.
    # They’re computed from true labels (y_test) and predicted probabilities (churn_probability)
    false_positive, true_positive, _ = roc_curve(y_test, churn_probability)

    roc_line = px.line(
        pd.DataFrame({"FPR": false_positive, "TPR": true_positive}),
        x="FPR", y="TPR",
        title=f"ROC Curve — Logistic Regression (AUC = {auc:.3f})",
        labels={"FPR":"False Positive Rate", "TPR":"True Positive Rate"}
    )
    roc_line.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    roc_line.update_layout(xaxis_range=[0,1], yaxis_range=[0,1])

    st.plotly_chart(roc_line, use_container_width=True)
    st.text('The model shows good ranking ability with AUC = 0.843, well above the dashed ' \
            'random baseline. The curve rises quickly at low false-positive rates: around FPR ≈ 10% the ' \
            'TPR (recall) is roughly ~60%, and by FPR ≈ 20% it’s around ~70–75%.')

    # Explainability
    # tabs
    st.subheader("Explainability")
    tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])

    with tab_shap:
        # sub-tabs INSIDE the SHAP tab
        sub_global, sub_local = st.tabs(["Global importance", "Per-customer"])

    # --- Global importance ---
    with sub_global:
        explainer = shap.TreeExplainer(random_forest_model)  # RF -> TreeExplainer

        # smaller sample to keep it fast
        X_sample = X_test.sample(min(300, len(X_test)), random_state=42)

        exp = explainer(X_sample)           # shap.Explanation
        vals = exp.values                   # numpy array

        # handle possible 3D output (rows, features, classes)
        if vals.ndim == 3:
            pos_idx = list(random_forest_model.classes_).index(1)
            vals = vals[:, :, pos_idx]

        avg_abs = np.abs(vals).mean(axis=0)
        importance_df = (
            pd.DataFrame({"feature": X_sample.columns, "avg_abs_contribution": avg_abs})
            .sort_values("avg_abs_contribution", ascending=False)
        )

        # BUILD A FIGURE, then plot it
        import plotly.express as px
        fig = px.bar(
            importance_df.head(15).iloc[::-1],
            x="avg_abs_contribution", y="feature", orientation="h",
            title="Top 15 Features by Global Importance (SHAP — Random Forest)",
            labels={"avg_abs_contribution":"Average |contribution|", "feature":"Feature"}
        )
        sub_global.plotly_chart(fig, use_container_width=True)
        sub_global.text('This chart ranks features by how much they influence churn predictions on average '
        '(magnitude only). Tenure is the strongest driver, followed by Internet service type (e.g., Fiber optic), ' \
        'contract length (especially Two year), Total/Monthly charges, and Payment method (Electronic check). ' \
        'Billing options (PaperlessBilling) and service add-ons (OnlineSecurity, TechSupport) also matter.')

    # --- Per-customer (example pattern) ---
    with sub_local:
        row_idx = 0
        x_row = X_test.iloc[[row_idx]]
        row_exp = explainer(x_row)
        row_vals = row_exp.values
        if row_vals.ndim == 3:
            pos_idx = list(random_forest_model.classes_).index(1)
            row_vals = row_vals[:, :, pos_idx]

        contrib_df = (
            pd.DataFrame({
                "feature": X_test.columns,
                "shap_value": row_vals[0],
                "feature_value": x_row.iloc[0].values
            })
            .assign(abs_=lambda d: d.shap_value.abs())
            .sort_values("abs_", ascending=False)
            .drop(columns="abs_")
        )

        fig_row = px.bar(
            contrib_df.head(12).iloc[::-1],
            x="shap_value", y="feature", orientation="h",
            title=f"Top 12 Feature Contributions — Row {row_idx}",
            labels={"shap_value":"Contribution (→ churn + / ← non-churn −)", "feature":"Feature"}
        )
        sub_local.plotly_chart(fig_row, use_container_width=True)
        sub_local.text('Bars show how each feature pushed this customer’s prediction: right = toward churn, ' \
        'left = toward non-churn. The largest pushes toward churn come from tenure and TotalCharges, with smaller ' \
        'positive effects from PaymentMethod_Electronic check, PhoneService, and some contract/internet indicators. ' \
        'The main factor reducing risk is InternetService_Fiber optic (left bar), with small reductions from SeniorCitizen '
        'and PaperlessBilling. Bar length = strength of influence; the signed contributions combine (with the baseline) to ' \
        'produce this customer’s final churn probability.')
        sub_local.dataframe(contrib_df.head(12), use_container_width=True)
        sub_local.text('Positive SHAP values push this prediction toward churn; negative values push away. For this ' \
                       'customer, the largest push toward churn comes from TotalCharges (+0.137).Smaller pushes come ' \
                       'from Electronic check (+0.054), PhoneService (+0.043), Contract_Two year (+0.043), ' \
                       'InternetService_No (+0.035), plus OnlineBackup/OnlineSecurity and Contract_One year (small positives). ' \
                        'The main factors reducing churn risk are InternetService_Fiber optic (−0.063) and, slightly, SeniorCitizen (−0.016).')

    #----------------------- LIME -----------------------
    with tab_lime:
            
        lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
             feature_names=X_train.columns.tolist(),
             class_names=["No churn","Churn"],
             discretize_continuous=True,
             mode="classification"
        )

        lime_explanation = lime_explainer.explain_instance(
            data_row=X_test.iloc[[0]].iloc[0].values,
            predict_fn=random_forest_model.predict_proba,
            num_features=10
         )

        lime_items = lime_explanation.as_list()  # list of (feature_or_rule, weight)
        lime_table = pd.DataFrame(lime_items, columns=["feature_or_rule", "lime_weight"])

        fig_lime_customer = px.bar(
            lime_table.iloc[::-1],
            x="lime_weight", y="feature_or_rule",
            orientation="h",
            title=f"LIME local explanation — row 0",
            labels={"lime_weight":"Local weight (→ churn + / ← non-churn −)", "feature_or_rule":"Feature / Rule"}
        )

        tab_lime.plotly_chart(fig_lime_customer, use_container_width=True)
        tab_lime.text('Bars show simple rules near this customer and their local weight: right = pushes toward churn, ' \
        'left = pushes toward non-churn.\nBiggest risk drivers: not on a 2-year contract (Contract_Two year ≤ −0.56) '
        'and short tenure (tenure ≤ −0.95).\nOther factors nudging risk up: low total charges (TotalCharges ≤ −0.83), ' \
        'not on a 1-year contract, has internet service (InternetService_No ≤ −0.53 ⇒ not “no internet”), electronic check ' \
        'payment, and signals consistent with no TechSupport/OnlineSecurity.\nRisk reducer: not on fiber optic '
        '(InternetService_Fiber optic ≤ −0.89) strongly pulls the prediction toward non-churn; StreamingMovies also reduces risk slightly.')
        tab_lime.dataframe(lime_table, use_container_width=True)
        tab_lime.text('Each rule is a simple condition near this customer; the weight shows its local effect (positive → pushes ' \
        'toward churn, negative → pushes toward non-churn).\nBiggest risk drivers: not on a 2-year contract '
        'and very short tenure (both strong positives).\nOther risk-increasing signals: lower total charges '
        '(early lifecycle), not on a 1-year contract, has internet service (vs. “no internet”), pays by electronic ' \
        'check, and no TechSupport/OnlineSecurity.\nRisk reducers: not using fiber-optic internet (largest negative)'
        ' and StreamingMovies (small negative).')

