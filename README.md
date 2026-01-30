🏦 German Credit Risk Analysis (Predictive AI)

📌 Project Overview
This project uses Machine Learning to solve a critical banking problem: **Automating Loan Eligibility. 
By analyzing historical data from 1,000 applicants, I built a Random Forest Classifier to predict whether a new applicant is a "Good" or "Bad" credit risk. This model helps reduce financial default by filtering out high-risk loans automatically.

📊 Key Business Insights
⁕ High Loans = High Risk: Applicants requesting higher credit amounts are statistically more likely to default.
⁕ Checking Account Impact: Customers with "little" or "moderate" checking accounts are higher risk than those with "rich" accounts.
⁕ Duration Risk: Loans with a duration exceeding 24 months show a sharp increase in default rates.

🛠️ Tech Stack
⁕ Language: Python 3.x
⁕ Libraries: Scikit-Learn (Machine Learning), Pandas (Data Manipulation), Seaborn (Visualization).
⁕ Model: Random Forest Classifier.

📈 Model Performance
The model achieved a 73% Accuracy rate on the test set.
⁕ Precision: 78% for Good Risk.
⁕ Recall: 87% for Good Risk (The model effectively captures safe applicants).

🖼️ Visualization
⁕ Analyzed the relationship between Credit Amount and Risk Status:
![Credit Risk Boxplot](credit_risk_graph.png)
(Note: Higher credit amounts correlate with 'Bad' risk outcomes)

🚀 How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn seaborn matplotlib
2. Run the script:
   python german_credit_risk.py
3. The script will:
   ​Load and clean the data.
​   Generate a Box Plot visualization.
​   Train the Random Forest model.
​   Print the Accuracy Report.

👤 Author
Priyanka Deshpande Data Analyst | Python | Excel
