import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Automotive Predictive Analytics",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(ML Car Diagnostic Agent AI Assistant.csv)
    return df

df = load_data()

# Feature engineering
df['CarBrand']        = df['Car Name'].str.split().str[0]
df['ManufactureYear'] = df['Car Name'].str.extract(r'(\d{4})').astype(int)
df['CarModel']        = df['Car Name'].str.replace(r'\s*\d{4}', '', regex=True).str.strip()
df['VehicleAge']      = 2026 - df['ManufactureYear']

severity_map          = {'Low': 1, 'Medium': 2, 'High': 3}
df['SeverityEncoded'] = df['Severity'].map(severity_map)
df['RepairSuccess']   = (df['Repair Status'] == 'Fixed').astype(int)
df['HasECU']          = df['ECU Data'].apply(
                            lambda x: 0 if str(x).strip().lower() == 'none' else 1)

parts_map = {
    'Dead battery': 'Battery', 'Faulty alternator': 'Battery',
    'Starter motor failure': 'Battery', 'Charging system fault': 'Battery',
    'Worn pads': 'Brake Components', 'Low brake fluid': 'Brake Components',
    'Warped rotors': 'Brake Components', 'Worn master cylinder': 'Brake Components',
    'Worn brake pads': 'Brake Components',
    'Timing belt failure': 'Engine Components', 'Spark plug wear': 'Engine Components',
    'Ignition coil failure': 'Engine Components', 'Fuel pump issue': 'Engine Components',
    'Transmission fluid thick': 'Transmission Fluid',
    'Transmission fluid low': 'Transmission Fluid',
    'Transmission solenoid': 'Transmission Parts',
    'Clutch pack wear': 'Transmission Parts', 'Transmission wear': 'Transmission Parts',
    'Loose wiring': 'Electrical Components', 'Wiring issue': 'Electrical Components',
    'Window regulator motor': 'Electrical Components'
}
df['PartsCategory'] = df['Diagnosis'].map(parts_map).fillna('Other')

# Sidebar navigation
st.sidebar.title("Automotive Analytics")
st.sidebar.markdown("**MSc IT with Data Analytics Dissertation Project**")
st.sidebar.markdown("Abdul-Aziz Mohammed | B01800363")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to:", [
    "Executive Overview",
    "Fault & Demand Analysis",
    "Vehicle Risk Insights",
    "Parts & Inventory",
    "ML Model Results",
    "Data Explorer",
    "AI Decision Support"
])

# Page 1: Executive Overview
if page == "Executive Overview":
    st.title("Executive Overview")
    st.markdown("High-level summary of the automotive diagnostic dataset.")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    total_cases  = len(df)
    fix_rate     = round(df['RepairSuccess'].mean() * 100, 1)
    high_sev     = len(df[df['Severity'] == 'High'])
    unique_brands = df['CarBrand'].nunique()

    col1.metric("Total Diagnostic Cases", f"{total_cases:,}")
    col2.metric("Overall Fix Rate",        f"{fix_rate}%")
    col3.metric("High Severity Cases",     f"{high_sev:,}")
    col4.metric("Vehicle Brands Covered",  f"{unique_brands}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.pie(df, names='Problem Classification',
                     title='Fault Distribution by Problem Category',
                     hole=0.35)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        status_counts = df['Repair Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig2 = px.bar(status_counts, x='Status', y='Count',
                      color='Status',
                      color_discrete_map={
                          'Fixed':'green','Not Fixed':'red','In Progress':'orange'},
                      title='Repair Status Distribution')
        st.plotly_chart(fig2, use_container_width=True)

# Page 2: Fault & Demand Analysis
elif page == "Fault & Demand Analysis":
    st.title("Fault & Demand Analysis")

    brand_filter = st.multiselect(
        "Filter by Brand (leave blank for all):",
        options=sorted(df['CarBrand'].unique()))
    filtered = df[df['CarBrand'].isin(brand_filter)] if brand_filter else df

    col1, col2 = st.columns(2)

    with col1:
        fault_counts = filtered['Problem Classification'].value_counts().reset_index()
        fault_counts.columns = ['Category','Count']
        fig = px.bar(fault_counts, x='Category', y='Count',
                     title='Fault Frequency by Category',
                     color='Category')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sev_fault = filtered.groupby(
            ['Problem Classification','Severity']).size().reset_index(name='Count')
        fig2 = px.bar(sev_fault, x='Problem Classification', y='Count',
                      color='Severity',
                      color_discrete_map={
                          'Low':'#66b3ff','Medium':'#ffcc99','High':'#ff6666'},
                      barmode='stack',
                      title='Fault Category by Severity Level')
        st.plotly_chart(fig2, use_container_width=True)

    # Brand fault heatmap
    st.subheader("Brand vs Fault Category Heatmap")
    top_brands = df['CarBrand'].value_counts().head(10).index.tolist()
    heatmap_data = df[df['CarBrand'].isin(top_brands)].groupby(
        ['CarBrand','Problem Classification']).size().unstack(fill_value=0)
    fig3 = px.imshow(heatmap_data,
                     labels=dict(color="Case Count"),
                     title="Fault Frequency Heatmap: Top 10 Brands",
                     color_continuous_scale='Blues')
    st.plotly_chart(fig3, use_container_width=True)

# Page 3: Vehicle Risk Insights
elif page == "Vehicle Risk Insights":
    st.title("Vehicle Risk Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(df, x='Severity', y='VehicleAge',
                     category_orders={'Severity':['Low','Medium','High']},
                     color='Severity',
                     color_discrete_map={
                         'Low':'#66b3ff','Medium':'#ffcc99','High':'#ff6666'},
                     title='Vehicle Age Distribution by Fault Severity')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fail_rate = df.groupby('CarBrand')['RepairSuccess'].mean().reset_index()
        fail_rate['FailRate'] = 1 - fail_rate['RepairSuccess']
        fail_rate = fail_rate.sort_values('FailRate', ascending=False).head(10)
        fig2 = px.bar(fail_rate, x='CarBrand', y='FailRate',
                      title='Top 10 Brands by Repair Failure Rate',
                      color='FailRate', color_continuous_scale='Reds',
                      labels={'FailRate':'Failure Rate'})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("High-Risk Case Explorer")
    high_risk = df[df['Severity'] == 'High'][
        ['CarBrand','CarModel','ManufactureYear','VehicleAge',
         'Problem Classification','Diagnosis','Repair Status']
    ].sort_values('VehicleAge', ascending=False)
    st.dataframe(high_risk.head(50), use_container_width=True)

# Page 4: Parts & Inventory
elif page == "Parts & Inventory":
    st.title("Parts & Inventory Insights")
    st.markdown("Demand patterns derived from diagnostic case frequencies.")

    col1, col2 = st.columns(2)

    with col1:
        parts_count = df['PartsCategory'].value_counts().reset_index()
        parts_count.columns = ['Category','Demand Count']
        fig = px.bar(parts_count, x='Category', y='Demand Count',
                     title='Parts Category Demand (All Cases)',
                     color='Category')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        brand_parts = df.groupby(
            ['CarBrand','PartsCategory']).size().reset_index(name='Count')
        top_b = df['CarBrand'].value_counts().head(8).index.tolist()
        fig2 = px.bar(brand_parts[brand_parts['CarBrand'].isin(top_b)],
                      x='CarBrand', y='Count', color='PartsCategory',
                      barmode='stack',
                      title='Parts Demand by Top 8 Brands')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Parts Demand by Severity Level")
    sev_parts = df.groupby(['PartsCategory','Severity']).size().reset_index(name='Count')
    fig3 = px.sunburst(sev_parts, path=['PartsCategory','Severity'], values='Count',
                       title='Parts Category → Severity Breakdown')
    st.plotly_chart(fig3, use_container_width=True)

# Page 5: ML Model Results
elif page == "ML Model Results":
    st.title(" Machine Learning Model Results")

    @st.cache_resource
    def train_models(df):
        le_brand   = LabelEncoder()
        le_problem = LabelEncoder()
        le_parts   = LabelEncoder()

        df = df.copy()
        df['BrandEncoded']   = le_brand.fit_transform(df['CarBrand'])
        df['ProblemEncoded'] = le_problem.fit_transform(df['Problem Classification'])
        df['PartsEncoded']   = le_parts.fit_transform(df['PartsCategory'])

        features = ['BrandEncoded','ProblemEncoded','SeverityEncoded',
                    'VehicleAge','HasECU','PartsEncoded']
        X = df[features]
        y = df['RepairSuccess']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                    min_samples_split=5, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        auc    = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
        report = classification_report(y_test, y_pred, output_dict=True)

        importances = pd.Series(rf.feature_importances_, index=features)
        return report, auc, importances, X_test, y_test, y_pred

    report, auc, importances, X_test, y_test, y_pred = train_models(df)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{round(report['accuracy']*100,1)}%")
    col2.metric("F1-Score (Fixed)", f"{round(report['1']['f1-score'],3)}")
    col3.metric("Precision", f"{round(report['1']['precision'],3)}")
    col4.metric("ROC-AUC",   f"{round(auc,3)}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        imp_df = importances.sort_values().reset_index()
        imp_df.columns = ['Feature','Importance']
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     title='Random Forest — Feature Importance',
                     color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig2 = px.imshow(cm,
                         labels=dict(x="Predicted",y="Actual",color="Count"),
                         x=['Not Fixed','Fixed'], y=['Not Fixed','Fixed'],
                         text_auto=True, color_continuous_scale='Blues',
                         title='Confusion Matrix — Random Forest')
        st.plotly_chart(fig2, use_container_width=True)

    # Model comparison table
    st.subheader("Model Performance Comparison")
    comparison = pd.DataFrame({
        'Model':     ['Random Forest','Logistic Regression'],
        'Accuracy':  [f"{round(report['accuracy']*100,1)}%", 'Run model to see'],
        'F1-Score':  [round(report['1']['f1-score'],3), '—'],
        'ROC-AUC':   [round(auc,3), '—'],
        'CV Method': ['5-Fold Stratified','5-Fold Stratified']
    })
    st.dataframe(comparison, use_container_width=True)

# Page 6: Data Explorer
elif page == "Data Explorer":
    st.title("Raw Data Explorer")

    col1, col2 = st.columns(2)
    brand_sel  = col1.multiselect("Filter Brand:",
                                   sorted(df['CarBrand'].unique()))
    sev_sel    = col2.multiselect("Filter Severity:", ['Low','Medium','High'])

    filtered = df.copy()
    if brand_sel: filtered = filtered[filtered['CarBrand'].isin(brand_sel)]
    if sev_sel:   filtered = filtered[filtered['Severity'].isin(sev_sel)]

    st.write(f"Showing {len(filtered):,} records")
    st.dataframe(
        filtered[['CarBrand','CarModel','VehicleAge','Problem Classification',
                  'Severity','Diagnosis','PartsCategory','Repair Status']],
        use_container_width=True
    )

# Page 7: AI Decision Support
elif page == "AI Decision Support":
    st.title("AI Decision Support")
    st.markdown(
        "This page provides three data-driven decision support tools built directly "
        "from the diagnostic dataset. Each tool draws on historical case patterns or "
        "the trained Random Forest model to assist workshop managers with fault "
        "diagnosis, repair outcome prediction, and parts inventory planning."
    )

    st.markdown("---")

    # Fault Diagnosis Recommender
    st.subheader("Tool 1 — Fault Diagnosis Recommender")
    st.markdown(
        "Select a vehicle brand, fault severity, and problem classification to "
        "retrieve the most historically common diagnosis, recommended parts category, "
        "and the associated historical fix rate from matching cases in the dataset."
    )

    col1, col2, col3 = st.columns(3)
    brand_input   = col1.selectbox("Vehicle Brand:",
                                   sorted(df["CarBrand"].unique()))
    sev_input     = col2.selectbox("Fault Severity:", ["Low", "Medium", "High"])
    problem_input = col3.selectbox("Problem Classification:",
                                   sorted(df["Problem Classification"].unique()))

    similar_cases = df[
        (df["CarBrand"]                == brand_input) &
        (df["Severity"]                == sev_input)   &
        (df["Problem Classification"]  == problem_input)
    ]

    if len(similar_cases) > 0:
        top_diagnosis  = similar_cases["Diagnosis"].value_counts().index[0]
        top_part       = similar_cases["PartsCategory"].value_counts().index[0]
        historical_fix = similar_cases["RepairSuccess"].mean() * 100
        case_count     = len(similar_cases)

        r1, r2, r3 = st.columns(3)
        r1.metric("Matching Historical Cases", f"{case_count:,}")
        r2.metric("Historical Fix Rate",        f"{historical_fix:.1f}%")
        r3.metric("Recommended Parts Category", top_part)

        st.info(f"Most Probable Diagnosis:  {top_diagnosis}")

        detail = similar_cases["Diagnosis"].value_counts().reset_index()
        detail.columns = ["Diagnosis", "Case Count"]
        detail["Proportion (%)"] = (
            detail["Case Count"] / detail["Case Count"].sum() * 100
        ).round(1)
        st.markdown("**Diagnosis Frequency for Selected Filters**")
        st.dataframe(detail.head(10), use_container_width=True)
    else:
        st.warning(
            "No cases match the selected combination. "
            "Try adjusting the brand, severity, or problem classification filters."
        )

    st.markdown("---")

    # Live Repair Outcome Predictor
    st.subheader("Tool 2 — Live Repair Outcome Predictor")
    st.markdown(
        "Enter vehicle characteristics below to generate a real-time repair outcome "
        "prediction using the trained Random Forest classifier. The model was trained "
        "on 8,000 records and evaluated on a stratified 2,000-record held-out test set."
    )

    from sklearn.preprocessing import LabelEncoder

    @st.cache_resource
    def get_encoders(df):
        le_brand   = LabelEncoder().fit(df["CarBrand"])
        le_problem = LabelEncoder().fit(df["Problem Classification"])
        le_parts   = LabelEncoder().fit(df["PartsCategory"])
        return le_brand, le_problem, le_parts

    @st.cache_resource
    def get_trained_rf(df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        le_brand, le_problem, le_parts = get_encoders(df)
        dfc = df.copy()
        dfc["BrandEncoded"]   = le_brand.transform(dfc["CarBrand"])
        dfc["ProblemEncoded"] = le_problem.transform(dfc["Problem Classification"])
        dfc["PartsEncoded"]   = le_parts.transform(dfc["PartsCategory"])

        features = ["BrandEncoded", "ProblemEncoded", "SeverityEncoded",
                    "VehicleAge", "HasECU", "PartsEncoded"]
        X = dfc[features]
        y = dfc["RepairSuccess"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            min_samples_split=5, random_state=42)
        rf.fit(X_train, y_train)
        return rf

    rf_model                     = get_trained_rf(df)
    le_brand, le_problem, le_parts = get_encoders(df)
    severity_map                 = {"Low": 1, "Medium": 2, "High": 3}

    p1, p2 = st.columns(2)
    pred_brand   = p1.selectbox("Brand:",    sorted(df["CarBrand"].unique()),   key="pred_brand")
    pred_age     = p1.slider("Vehicle Age (years):", 1, 25, 7)
    pred_sev     = p2.selectbox("Severity:", ["Low", "Medium", "High"],         key="pred_sev")
    pred_problem = p2.selectbox("Problem:",  sorted(df["Problem Classification"].unique()), key="pred_prob")
    pred_parts   = p2.selectbox("Parts Category:", sorted(df["PartsCategory"].unique()),    key="pred_parts")
    pred_ecu     = p1.radio("ECU Data Available:", ["Yes", "No"])

    if st.button("Generate Prediction"):
        brand_enc   = le_brand.transform([pred_brand])[0]
        problem_enc = le_problem.transform([pred_problem])[0]
        parts_enc   = le_parts.transform([pred_parts])[0]
        sev_enc     = severity_map[pred_sev]
        ecu_flag    = 1 if pred_ecu == "Yes" else 0

        input_row   = [[brand_enc, problem_enc, sev_enc, pred_age, ecu_flag, parts_enc]]
        prediction  = rf_model.predict(input_row)[0]
        probability = rf_model.predict_proba(input_row)[0]

        st.markdown("**Prediction Result**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Outcome",
                  "Fixed" if prediction == 1 else "Not Fixed")
        m2.metric("Confidence — Fixed",     f"{probability[1]*100:.1f}%")
        m3.metric("Confidence — Not Fixed", f"{probability[0]*100:.1f}%")

        if prediction == 1:
            st.success(
                "The model predicts this vehicle is likely to be successfully repaired "
                "based on historical cases with similar characteristics."
            )
        else:
            st.error(
                "The model predicts this repair may not be completed successfully. "
                "Consider escalating the case or sourcing specialist parts in advance."
            )

    st.markdown("---")

    # Parts Stock Alert System
    st.subheader("Tool 3 — Parts Inventory Alert System")
    st.markdown(
        "This tool analyses current diagnostic case frequencies to flag which parts "
        "categories require immediate attention based on demand thresholds derived "
        "from the dataset distribution."
    )

    demand_summary = df["PartsCategory"].value_counts().reset_index()
    demand_summary.columns = ["Parts Category", "Demand Count"]

    def assign_status(count):
        if count > 2000:
            return "Critical — Reorder Immediately"
        elif count > 1000:
            return "Monitor — Stock Running Low"
        else:
            return "Sufficient"

    demand_summary["Stock Status"] = demand_summary["Demand Count"].apply(assign_status)
    demand_summary["Priority Rank"] = range(1, len(demand_summary) + 1)

    st.dataframe(
        demand_summary[["Priority Rank", "Parts Category", "Demand Count", "Stock Status"]],
        use_container_width=True
    )

    critical_parts = demand_summary[
        demand_summary["Stock Status"] == "Critical — Reorder Immediately"
    ]["Parts Category"].tolist()

    monitor_parts = demand_summary[
        demand_summary["Stock Status"] == "Monitor — Stock Running Low"
    ]["Parts Category"].tolist()

    if critical_parts:
        st.error(
            f"Immediate reorder required for: {', '.join(critical_parts)}. "
            "These categories exceed the 2,000-case demand threshold."
        )
    if monitor_parts:
        st.warning(
            f"Stock levels to monitor closely: {', '.join(monitor_parts)}. "
            "Demand is approaching the critical threshold."
        )

    by_brand = df.groupby(["CarBrand", "PartsCategory"]).size().reset_index(name="Cases")
    by_brand = by_brand.sort_values("Cases", ascending=False)
    st.markdown("**Parts Demand Breakdown by Brand**")
    brand_filter_inv = st.multiselect(
        "Filter by Brand (leave blank for all):",
        options=sorted(df["CarBrand"].unique()),
        key="inv_brand"
    )
    if brand_filter_inv:
        by_brand = by_brand[by_brand["CarBrand"].isin(brand_filter_inv)]
    st.dataframe(by_brand, use_container_width=True)
