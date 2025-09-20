
import streamlit as st
from io import BytesIO
import pandas as pd
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(page_title="FinTrack - Personal Finance Analyzer", layout="wide")

st.title("FinTrack - Personal Finance & Expense Analyzer")
st.markdown("Take control of your money—track, analyze, and predict your expenses in a few clicks!")
st.markdown("I personally face difficulty in managing expenses.So, I designed this project using data science and full stack development to track and manage expenses. Analyze your income, expenses, and savings from CSV/Excel files or Bank Statements with interactive charts and predictions.")

uploaded_file = st.file_uploader("Upload your bank statement (CSV/Excel)", type=["csv", "xlsx"])

def load_data(file):
    if file is not None:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    elif os.path.exists("data/transactions_3months.xlsx"):
        return pd.read_excel("data/transactions_3months.xlsx")
    else:
        return None

df = load_data(uploaded_file)

if df is None:
    st.warning("Please upload a dataset or place `transactions_3months.xlsx` inside a `data/` folder.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df["Type"] = df["Type"].str.title().str.strip()

categories = {
    "Food": ["Dominos", "Zomato", "Pizza", "Restaurant", "Cafe"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Shopping"],
    "Bills": ["Electricity", "Internet", "Insurance", "Bill", "DTH"],
    "Travel": ["Uber", "Ola", "Flight", "Train", "Bus", "Taxi"],
    "Groceries": ["Grocery", "Supermarket", "Big Bazaar", "DMart"],
    "Rent": ["Rent"],
    "Entertainment": ["Movie", "Netflix", "Spotify", "BookMyShow"],
    "Income": ["Salary", "Freelance", "Credit", "Transfer"],
    "Healthcare": ["Clinic", "Hospital", "Pharmacy", "Doctor"]
}

def categorize(desc):
    for cat, keys in categories.items():
        if any(k.lower() in str(desc).lower() for k in keys):
            return cat
    return "Other"

df["Category"] = df["Description"].apply(categorize)

st.sidebar.header("Filters")

category_filter = st.sidebar.multiselect("Select Categories", options=df["Category"].unique(), default=df["Category"].unique())
filtered_df = df[df["Category"].isin(category_filter)]

if filtered_df['Date'].notna().any():
    min_date = filtered_df['Date'].min()
    max_date = filtered_df['Date'].max()
else:
    min_date = max_date = pd.to_datetime("today")

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
if len(date_range) == 2 and filtered_df['Date'].notna().any():
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

search_text = st.sidebar.text_input("Search Description/Notes")
if search_text:
    filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search_text, case=False).any(), axis=1)]

st.subheader("Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Entries", len(filtered_df))
if 'Amount' in filtered_df.columns:
    col2.metric("Total Amount", f"₹{filtered_df['Amount'].sum():,.2f}")
col3.metric("Categories Shown", filtered_df['Category'].nunique())

tab1, tab2, tab3, tab4 = st.tabs(["Transactions", "Analytics", "Insights", "Predictions"])

with tab1:
    st.subheader("Transactions")
    st.dataframe(filtered_df, use_container_width=True)

with tab2:
    st.subheader("Expense Breakdown")
    expense_df = filtered_df[filtered_df["Type"] == "Debit"]
    category_summary = expense_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    
    if not category_summary.empty:
        fig = px.bar(category_summary, x=category_summary.index, y=category_summary.values,
                     title="Expenses by Category", color=category_summary.index)
        st.plotly_chart(fig, use_container_width=True)

    
    if not category_summary.empty:
        pie_fig = px.pie(expense_df, names="Category", values="Amount", title="Category-wise Expense Distribution", hole=0.3)
        st.plotly_chart(pie_fig, use_container_width=True)

    
    st.subheader("Monthly Spending Trend")
    if filtered_df['Date'].notna().any():
        filtered_df["Month"] = filtered_df["Date"].dt.to_period("M")
        monthly_summary = filtered_df.groupby("Month")["Amount"].sum()
        fig2 = px.line(monthly_summary, x=monthly_summary.index.astype(str), y=monthly_summary.values,
                       title="Monthly Transactions Trend", markers=True)
        st.plotly_chart(fig2, use_container_width=True)


with tab3:
    st.subheader("Financial Insights")
    total_income = filtered_df[filtered_df["Type"] == "Credit"]["Amount"].sum()
    total_expenses = filtered_df[filtered_df["Type"] == "Debit"]["Amount"].sum()
    savings = total_income - total_expenses

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"₹{total_income:,.0f}")
    col2.metric("Total Expenses", f"₹{total_expenses:,.0f}")
    col3.metric("Net Savings", f"₹{savings:,.0f}")

    if savings < 0:
        st.error("You're spending more than you earn. Time to review recurring expenses!")
    elif savings < 5000:
        st.warning("Low savings — cut down on Food & Entertainment.")
    else:
        st.success("Good job! You are saving a healthy amount.")

    st.subheader("Top 5 Spending Categories")
    st.table(category_summary.head(5))

with tab4:
    st.subheader("Expense Prediction (Next Month)")

    if filtered_df['Date'].notna().any() and 'Amount' in filtered_df.columns:
        filtered_df["Month_Num"] = filtered_df["Date"].dt.month
        monthly_expense = filtered_df[filtered_df["Type"] == "Debit"].groupby("Month_Num")["Amount"].sum().reset_index()

        if len(monthly_expense) >= 2:
            X = monthly_expense[["Month_Num"]]
            y = monthly_expense["Amount"]

            model = LinearRegression()
            model.fit(X, y)

            next_month = max(monthly_expense["Month_Num"]) + 1
            y_pred = model.predict([[next_month]])[0]

            st.write(f"Predicted Expenses for Month {next_month}: ₹{y_pred:,.0f}")

            # Plot actual vs prediction
            fig3 = px.line(monthly_expense, x="Month_Num", y="Amount", markers=True, title="Actual Expenses vs Prediction")
            fig3.add_scatter(x=[next_month], y=[y_pred], mode="markers+text", name="Predicted",
                             text=[f"Pred: ₹{int(y_pred)}"], textposition="top center")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough months of data for prediction. Upload at least 2 months of data.")


def to_excel(df):
    output = BytesIO()
    # Use 'with' to automatically save and close
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='FilteredData')
    processed_data = output.getvalue()
    return processed_data

excel_data = to_excel(filtered_df)
st.download_button(
    label="📥 Download Filtered Data as Excel",
    data=excel_data,
    file_name='filtered_data.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
