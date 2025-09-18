import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💰", layout="wide")

# Title and description
st.title("🏦 Financial Transaction Fraud Analysis & Prediction")
st.markdown("Analyze transaction patterns, detect fraudulent activities, and calculate financial impact")

# Load data from CSV file
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'/mount/src/data-science/Projects/fraud_detection/Fraud_Analysis_Dataset.csv')
        return df
    except FileNotFoundError:
        st.error("❌ File 'Fraud_Analysis_Dataset.csv' not found. Please make sure the file is in the same directory.")
        st.write("Current working directory:", os.getcwd())
        st.write("Files in directory:", os.listdir())
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()

# Load the data
df = load_data()

# Try to load model if available
try:
    model = joblib.load('fraud_detection_pipeline.pkl')
    model_available = True
except FileNotFoundError:
    st.warning("⚠️ Model file 'fraud_detection_pipeline.pkl' not found. Prediction features will be limited.")
    model_available = False
except Exception as e:
    st.warning(f"⚠️ Error loading model: {e}. Prediction features will be limited.")
    model_available = False

# Financial impact calculation function
def calculate_financial_impact(true_positives, false_positives, false_negatives, avg_transaction_amount, 
                              fraud_prevention_cost_per_case=25, operational_cost_savings=0.15):
    """
    Calculate financial impact of fraud detection
    """
    # Revenue protection from prevented fraud
    revenue_protected = true_positives * avg_transaction_amount
    
    # Cost of false positives (investigation costs)
    fp_investigation_cost = false_positives * fraud_prevention_cost_per_case
    
    # Losses from undetected fraud
    losses_from_fraud = false_negatives * avg_transaction_amount
    
    # Operational efficiency savings (reduced manual review)
    operational_savings = (true_positives + false_positives) * avg_transaction_amount * operational_cost_savings
    
    # Net financial impact
    net_impact = revenue_protected + operational_savings - fp_investigation_cost - losses_from_fraud
    
    return {
        'revenue_protected': revenue_protected,
        'fp_investigation_cost': fp_investigation_cost,
        'losses_from_fraud': losses_from_fraud,
        'operational_savings': operational_savings,
        'net_impact': net_impact,
        'roi': (net_impact / (fp_investigation_cost + losses_from_fraud)) * 100 if (fp_investigation_cost + losses_from_fraud) > 0 else 0
    }

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Data Visualization", "🔮 Fraud Prediction", "💵 Financial Impact"])

with tab1:
    # Data Visualization Section
    st.header("📈 Transaction Data Analysis")
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Fraudulent Transactions", f"{df['isFraud'].sum():,}")
    with col3:
        st.metric("Fraud Rate", f"{df['isFraud'].mean()*100:.4f}%")
    with col4:
        st.metric("Total Amount", f"${df['amount'].sum():,.2f}")
    
    # Transaction type distribution
    if 'type' in df.columns:
        st.subheader("Transaction Type Distribution")
        type_counts = df['type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                     title="Distribution of Transaction Types")
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud by type
    if 'type' in df.columns:
        st.subheader("Fraud by Transaction Type")
        fraud_by_type = df[df['isFraud'] == 1]['type'].value_counts()
        fig = px.bar(x=fraud_by_type.index, y=fraud_by_type.values, 
                     title="Fraudulent Transactions by Type",
                     labels={'x': 'Transaction Type', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Amount distribution
    st.subheader("Transaction Amount Distribution")
    fig = px.histogram(df, x='amount', nbins=50, title="Distribution of Transaction Amounts")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Fraud Prediction Section
    st.header("🔮 Fraud Prediction")
    
    if model_available:
        st.markdown('Please enter the transaction details and use the predict button')
        st.divider()
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Input fields
            transaction_type = st.selectbox('Transaction Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
            
            oldbalanceOrg = st.number_input('Old Balance (Sender)', min_value=0.0, value=1000.0, step=100.0)
            newbalanceOrig = st.number_input('New Balance (Sender)', min_value=0.0, value=900.0, step=100.0)
        
        with col2:
            amount = st.number_input('Amount', min_value=0.0, value=1000.0, step=100.0)
            oldbalanceDest = st.number_input('Old Balance (Receiver)', min_value=0.0, value=0.0, step=100.0)
            newbalanceDest = st.number_input('New Balance (Receiver)', min_value=0.0, value=1000.0, step=100.0)
            
            # Predict button
        if st.button('Predict Fraud', type='primary', use_container_width=False):
            input_data = pd.DataFrame([{
                'type': transaction_type,
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest
            }])
                
            try:
                # Get prediction and probability
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                    
                # Display prediction
                st.subheader("Prediction Result")
                    
                if prediction == 1:
                    st.error(f"⚠️ Fraud Alert: This transaction is predicted to be fraudulent (Probability: {probability[1]*100:.2f}%)")
                else:
                    st.success(f"✅ Legitimate Transaction: This transaction appears to be legitimate (Probability: {probability[0]*100:.2f}%)")
                    
                # Show probability distribution
                fig = go.Figure(data=[
                    go.Bar(x=['Legitimate', 'Fraudulent'], 
                           y=[probability[0]*100, probability[1]*100],
                           marker_color=['green', 'red'])
                ])
                fig.update_layout(title='Prediction Probability',
                                yaxis_title='Probability (%)')
                st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    else:
        st.warning("Model not available. Please ensure the model file is in the correct location.")

with tab3:
    st.header("💵 Financial Impact Analysis")
    st.markdown("Calculate the financial impact of fraud detection and prevention")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        # Display basic info about the dataset
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(df))
        col2.metric("Fraudulent Transactions", df['isFraud'].sum())
        col3.metric("Fraud Rate", f"{df['isFraud'].mean()*100:.2f}%")
        
        # Show a sample of the data
        with st.expander("View Data Sample"):
            st.dataframe(df.head())
        
        # Calculate financial metrics
        st.subheader("Financial Impact Analysis")
        
        # Calculate key metrics
        total_revenue = df[df['isFraud'] == 0]['amount'].sum()
        total_fraud_loss = df[df['isFraud'] == 1]['amount'].sum()
        prevented_loss = total_fraud_loss  # Assuming all detected fraud is prevented
        net_profit = total_revenue - total_fraud_loss
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("Total Fraud Loss", f"${total_fraud_loss:,.2f}")
        col3.metric("Prevented Loss", f"${prevented_loss:,.2f}")
        col4.metric("Net Profit", f"${net_profit:,.2f}")
        
        # Visualization 1: Financial impact chart
        st.subheader("Financial Impact Visualization")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        labels = ['Revenue', 'Fraud Loss']
        sizes = [total_revenue, total_fraud_loss]
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Revenue vs Fraud Loss')
        
        # Bar chart
        categories = ['Revenue', 'Fraud Loss', 'Net Profit']
        values = [total_revenue, total_fraud_loss, net_profit]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        ax2.bar(categories, values, color=colors)
        ax2.set_title('Financial Impact')
        ax2.ticklabel_format(style='plain', axis='y')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        st.pyplot(fig)
        
        # Fraud by transaction type
        st.subheader("Fraud Analysis by Transaction Type")
        
        if 'type' in df.columns:
            fraud_by_type = df[df['isFraud'] == 1].groupby('type')['amount'].agg(['count', 'sum'])
            fraud_by_type.columns = ['Transaction Count', 'Total Amount']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Fraud Statistics by Transaction Type")
                st.dataframe(fraud_by_type)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                fraud_by_type['Total Amount'].plot(kind='bar', color='#e74c3c', ax=ax)
                ax.set_title('Fraud Amount by Transaction Type')
                ax.ticklabel_format(style='plain', axis='y')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.warning("Transaction type data not available for analysis")
        
        # Advanced metrics
        st.subheader("Advanced Financial Metrics")
        
        # Calculate additional metrics
        avg_fraud_amount = df[df['isFraud'] == 1]['amount'].mean()
        max_fraud_amount = df[df['isFraud'] == 1]['amount'].max()
        fraud_to_revenue_ratio = total_fraud_loss / total_revenue
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg. Fraud Amount", f"${avg_fraud_amount:,.2f}")
        col2.metric("Max Fraud Amount", f"${max_fraud_amount:,.2f}")
        col3.metric("Fraud/Revenue Ratio", f"{fraud_to_revenue_ratio*100:.4f}%")
        col4.metric("ROI of Prevention", f"{(prevented_loss/(prevented_loss*0.1))*100:.2f}%")  # Assuming 10% cost of prevention
        
        # Savings calculation
        st.subheader("Potential Savings Analysis")
        
        prevention_cost = prevented_loss * 0.1  # Assuming 10% cost of prevention
        net_savings = prevented_loss - prevention_cost
        
        savings_data = pd.DataFrame({
            'Category': ['Prevented Loss', 'Prevention Cost', 'Net Savings'],
            'Amount': [prevented_loss, prevention_cost, net_savings]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(savings_data['Category'], savings_data['Amount'], 
               color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_title('Fraud Prevention Savings Analysis')
        ax.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig)
        
    else:
        st.info("Please upload a CSV file to begin the analysis.")
        
        # Show sample calculations
        st.subheader("How This Analysis Works")
        
        st.markdown("""
        This application calculates financial metrics based on your fraud detection dataset:
        
        1. **Total Revenue**: Sum of all legitimate transactions (isFraud = 0)
        2. **Total Fraud Loss**: Sum of all fraudulent transactions (isFraud = 1)
        3. **Prevented Loss**: Assuming all detected fraud is prevented (same as Total Fraud Loss)
        4. **Net Profit**: Revenue minus fraud losses
        
        The application also provides:
        - Visualization of financial impact
        - Analysis of fraud by transaction type
        - Advanced metrics like average fraud amount
        - Savings analysis from fraud prevention
        """)


# Footer
st.divider()
st.caption("""
**Financial Impact Analysis Notes:**
- Calculations based on actual transaction data and fraud indicators
- Prevented loss assumes all detected fraud is successfully prevented
- ROI calculation considers prevention costs as 10% of prevented losses
- Customizable parameters for different business scenarios
""")