import base64
import io
import os
import time
from datetime import datetime
from statistics import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import sympy as sp
from jax._src.source_info_util import summarize
from openai import OpenAI
from pygments.lexers import go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
# SETUP
st.set_page_config(page_title="Smart Engineering Project Manager", layout="wide")
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

st.title("🚀 Smart Engineering Project Manager (SEPM)")

menu = st.sidebar.selectbox("Navigate",
                            ["🏠 Home", "📁 Project Manager", "📐 Engineering Toolkit", "📊 Gantt Chart", "📄 Report Viewer",
                             "🤖 AI Assistant", "📈 Data Analyzer", "📋 BOM Manager", "🧠 Calculator",
                             "📘 Engineering Glossary", "📎 Notes & Docs", "🔁 Version Control", "🛠️ Unit Converter",
                             "📦 Inventory Tracker", "🎯 Goal Tracker", "🧪 Test Log", "🕒 Time Tracker",
                             "🖼️ CAD Viewer", "🔍 Error Log Analyzer", "📡 Sensor Monitor", "🧰 Tool Scheduler",
                             "🎵📏 Sound metre pro", "📝✨ Notes summarizer", "🎥🔧 Engineering videos"])


# HOME
def ARIMA():
    pass


if menu == "🏠 Home":
    placeholder = st.empty()
    message = "### Welcome to SEPM — Your All-in-One Engineering Workflow App!"
    typed_message = ""
    for char in message:
        typed_message += char
        placeholder.markdown(typed_message)
        time.sleep(0.05)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📂 Active Projects", len(st.session_state.tasks))
    with col2:
        st.metric("📎 Files Uploaded", len(st.session_state.uploaded_files))
    with col3:
        st.metric("⚙️ Tools Used", 24)
    st.info("Use the sidebar to upload tasks, calculate equations, visualize reports, and get help from AI.")


# PROJECT MANAGER
elif menu == "📁 Project Manager":
    st.subheader("📁 Project Manager")

    # Initialize tasks and uploads
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    # Task Form
    with st.form("task_form"):
        name = st.text_input("📝 Task Name")
        deadline = st.date_input("📅 Deadline")
        status = st.selectbox("📌 Status", ["Not Started", "In Progress", "Completed"])
        submitted = st.form_submit_button("➕ Add Task")

        if submitted and name:
            task_id = len(st.session_state.tasks) + 1
            st.session_state.tasks.append({
                "ID": task_id,
                "Task Name": name,
                "Deadline": deadline,
                "Status": status
            })
            st.success(f"✅ Task '{name}' added!")

    # Task Table
    if st.session_state.tasks:
        st.markdown("### 📋 Current Task List")

        task_df = pd.DataFrame(st.session_state.tasks)


        # Color-code status
        def highlight_status(val):
            color = {
                "Completed": "#b6fcb6",
                "In Progress": "#fff3cd",
                "Not Started": "#f8d7da"
            }.get(val, "white")
            return f'background-color: {color}'


        # Check for overdue or due today
        today = pd.to_datetime("today").normalize()
        task_df["Deadline"] = pd.to_datetime(task_df["Deadline"])
        task_df["Due Alert"] = task_df["Deadline"].apply(
            lambda d: "🔴 Overdue" if d < today else "🟡 Due Today" if d == today else "✅ Upcoming"
        )

        st.dataframe(task_df.style.applymap(highlight_status, subset=["Status"]))

        # Task Summary Pie Chart
        st.markdown("### 📊 Task Progress Summary")
        status_counts = task_df["Status"].value_counts()
        st.pyplot(status_counts.plot.pie(autopct="%1.1f%%", ylabel="").figure)

        # Remove task
        remove_task = st.selectbox("🗑️ Select Task to Remove", options=[t["Task Name"] for t in st.session_state.tasks])
        if st.button("Remove Selected Task"):
            st.session_state.tasks = [t for t in st.session_state.tasks if t["Task Name"] != remove_task]
            st.success(f"❌ Task '{remove_task}' removed.")

    # File Upload
    st.markdown("### 📤 Upload Project Files")
    files = st.file_uploader("Choose files", accept_multiple_files=True)

    if files:
        st.session_state.uploaded_files += files
        st.success("✅ Files uploaded!")

    if st.session_state.uploaded_files:
        st.markdown("### 📂 Uploaded Files")
        for file in st.session_state.uploaded_files:
            st.write(f"📄 {file.name}")
            if file.type.startswith("image/"):
                st.image(file)
            elif file.type.startswith("text/"):
                content = file.read().decode("utf-8")
                st.text_area(f"Preview of {file.name}", content, height=150)


# TOOLKIT
elif menu == "📐 Engineering Toolkit":
    st.subheader("🧮 Engineering Calculator (Symbolic + Numeric)")
    calc_type = st.radio("Choose Mode:", ["Symbolic", "Numeric"])

    x = sp.symbols("x")  # General symbolic variable for SymPy

    if calc_type == "Symbolic":
        st.markdown("### ✳️ Symbolic Math")
        expr = st.text_input("Enter expression (e.g., x**3 + 2*x + 1)")
        operation = st.selectbox("Choose Operation", ["Differentiate", "Integrate", "Solve Equation", "Taylor Series"])

        var = st.text_input("Variable (default: x)", value="x")

        if st.button("🔍 Compute Symbolically"):
            try:
                sym_var = sp.Symbol(var)
                parsed_expr = sp.sympify(expr)

                if operation == "Differentiate":
                    result = sp.diff(parsed_expr, sym_var)
                    st.latex(f"\\frac{{d}}{{d{var}}}({sp.latex(parsed_expr)}) = {sp.latex(result)}")

                elif operation == "Integrate":
                    result = sp.integrate(parsed_expr, sym_var)
                    st.latex(f"\\int {sp.latex(parsed_expr)}\,d{var} = {sp.latex(result)} + C")

                elif operation == "Solve Equation":
                    solutions = sp.solve(parsed_expr, sym_var)
                    for i, sol in enumerate(solutions, 1):
                        st.write(f"Solution {i}: {sol}")

                elif operation == "Taylor Series":
                    order = st.slider("Select Taylor Series Order", 1, 10, 4)
                    taylor = sp.series(parsed_expr, sym_var, 0, order).removeO()
                    st.latex(f"{sp.latex(parsed_expr)} \\approx {sp.latex(taylor)}")
                    st.info("Expansion around 0 (Maclaurin Series)")

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

    else:
        st.markdown("### 📈 Graph Plotting")

        f_input = st.text_area("Enter function(s) to plot (separate multiple with commas)", "sin(x), cos(x)")
        domain = st.slider("Select x-range", -50, 50, (-10, 10))
        show_values = st.checkbox("Show evaluated values", value=False)

        if st.button("📊 Plot Function(s)"):
            try:
                x_vals = np.linspace(domain[0], domain[1], 500)
                fig, ax = plt.subplots()
                func_list = [f.strip() for f in f_input.split(",")]

                for func in func_list:
                    y_vals = eval(f"np.{func}")
                    ax.plot(x_vals, y_vals, label=f"${func}$")

                    if show_values:
                        st.write(f"**Function:** `{func}`")
                        st.write(pd.DataFrame({
                            "x": np.round(x_vals, 2),
                            "y": np.round(y_vals, 4)
                        }).head(10))

                ax.set_title("Function Plot")
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ Error evaluating function(s): {e}")


# GANTT CHART
elif menu == "📊 Gantt Chart":
    st.subheader("📅 Auto-Generated Gantt Chart with Progress Tracking")

    if st.session_state.tasks:
        df = pd.DataFrame(st.session_state.tasks)

        # Convert deadlines to datetime
        df['End'] = pd.to_datetime(df['deadline'])
        df['Start'] = datetime.today()

        # Add Duration column
        df['Duration (days)'] = (df['End'] - df['Start']).dt.days.clip(lower=1)

        # Map status to colors
        status_colors = {
            "Not Started": "red",
            "In Progress": "orange",
            "Completed": "green"
        }
        df['Color'] = df['status'].map(status_colors)

        # Build Gantt chart with Plotly
        fig = go.Figure()

        for i, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Duration (days)']],
                y=[row['name']],
                base=row['Start'],
                orientation='h',
                marker_color=row['Color'],
                name=row['status'],
                hovertemplate=(
                    f"<b>Task:</b> {row['name']}<br>"
                    f"<b>Status:</b> {row['status']}<br>"
                    f"<b>Start:</b> {row['Start'].strftime('%Y-%m-%d')}<br>"
                    f"<b>End:</b> {row['End'].strftime('%Y-%m-%d')}<br>"
                    f"<b>Duration:</b> {row['Duration (days)']} day(s)"
                )
            ))

        fig.update_layout(
            title="📊 Project Timeline",
            xaxis_title="Date",
            yaxis_title="Task Name",
            barmode='stack',
            height=50 + len(df) * 40,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=120, r=40, t=50, b=40),
        )

        fig.update_xaxes(type='date')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ No tasks have been added yet. Please add tasks under the '📁 Project Manager' section.")

# REPORT VIEWER
elif menu == "📄 Report Viewer":
    st.subheader("📄 Report/Simulation Viewer")
    uploaded = st.file_uploader("Upload a CSV, PDF, or TXT file", type=["csv", "pdf", "txt"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            st.dataframe(df)
        elif uploaded.name.endswith(".txt"):
            content = uploaded.read().decode()
            st.text_area("Contents", content, height=300)
        elif uploaded.name.endswith(".pdf"):
            base64_pdf = base64.b64encode(uploaded.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

# AI ASSISTANT
elif menu == "🤖 AI Assistant":
    st.subheader("💬 Ask the Engineering Assistant (GPT-powered)")

    if OPENAI_API_KEY is None:
        st.error("AI Assistant is unavailable due to missing API key.")
    else:
        question = st.text_area("Ask a technical engineering question:")

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking... 🤖"):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system",
                                 "content": "You are a smart engineering assistant who explains complex concepts "
                                            "clearly."},
                                {"role": "user", "content": question},
                            ],
                            max_tokens=500,
                            temperature=0.3
                        )
                        answer = response.choices[0].message.content
                        st.markdown("### Answer:")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Oops, something went wrong: {e}")
elif menu == "📈 Data Analyzer":
    st.subheader("📊 Advanced Data Analyzer & ML Toolkit")

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head())

        st.markdown("### 📋 Basic Info")
        st.write(f"**Shape:** {df.shape}")
        st.write("**Columns:**", df.columns.tolist())
        st.write("**Data Types:**")
        st.write(df.dtypes)

        st.markdown("### 📊 Descriptive Statistics")
        st.write(df.describe())

        st.markdown("### 🧩 Missing Data Summary")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Missing Count", "index": "Column"}))

        # Column selection
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        col_x = st.selectbox("📌 Select X-axis", numeric_cols)
        col_y = st.selectbox("📌 Select Y-axis (Optional)", ["None"] + numeric_cols)

        st.markdown("### 📈 Visualization")
        plot_type = st.selectbox("Choose plot type",
                                 ["Histogram", "Box Plot", "Line Chart", "Scatter Plot", "Correlation Heatmap"])

        if plot_type == "Histogram":
            fig = px.histogram(df, x=col_x)
            st.plotly_chart(fig)
        elif plot_type == "Box Plot":
            fig = px.box(df, y=col_x)
            st.plotly_chart(fig)
        elif plot_type == "Line Chart":
            fig = px.line(df, x=df.index, y=col_x)
            st.plotly_chart(fig)
        elif plot_type == "Scatter Plot":
            if col_y != "None":
                fig = px.scatter(df, x=col_x, y=col_y)
                st.plotly_chart(fig)
            else:
                st.warning("Select Y-axis for scatter plot.")
        elif plot_type == "Correlation Heatmap":
            fig = px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig)

        st.markdown("### 🔎 Custom Query")
        query = st.text_input("Enter a pandas query (e.g., Age > 30 and Salary < 50000)")
        if query:
            try:
                filtered_df = df.query(query)
                st.dataframe(filtered_df)
                st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False),
                                   file_name="filtered_data.csv")
            except Exception as e:
                st.error(f"Query Error: {e}")

        # ML SECTION
        st.markdown("### 🤖 Basic Machine Learning")
        ml_mode = st.selectbox("Select ML Task", ["None", "Regression", "Classification", "Clustering (KMeans)",
                                                  "Time Series Forecasting"])

        if ml_mode != "None":
            target = st.selectbox("🎯 Select Target Variable", df.columns)
            if ml_mode in ["Regression", "Classification"]:
                X = df.drop(columns=[target])
                y = df[target]
                X = X.select_dtypes(include=np.number).dropna()
                y = y.loc[X.index]

                if len(X) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    if ml_mode == "Regression":
                        model = LinearRegression()
                    else:
                        model = RandomForestClassifier()

                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    y_pred = model.predict(X_test)

                    st.write("**Model Accuracy / R² Score:**", score)
                    st.write("**Predictions Sample:**")
                    st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).head())
                else:
                    st.warning("Insufficient data for ML modeling.")

            elif ml_mode == "Clustering (KMeans)":
                X = df[numeric_cols].dropna()
                n_clusters = st.slider("Select number of clusters", 2, 10, 3)
                kmeans = KMeans(n_clusters=n_clusters)
                df['Cluster'] = kmeans.fit_predict(X)
                st.success("Clustering Completed!")
                st.dataframe(df[['Cluster'] + numeric_cols[:2]])
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=df['Cluster'].astype(str))
                st.plotly_chart(fig)

            elif ml_mode == "Time Series Forecasting":
                date_col = st.selectbox("Select Date Column", df.columns)
                value_col = st.selectbox("Select Value Column", numeric_cols)
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(by=date_col)
                series = df.set_index(date_col)[value_col].dropna()
                try:
                    model = ARIMA()
                    results = model.fit()
                    forecast = results.forecast(steps=10)
                    st.write("**10-Step Forecast:**")
                    st.line_chart(forecast)
                except Exception as e:
                    st.error(f"ARIMA Forecasting Error: {e}")
    else:
        st.info("📁 Please upload a CSV file to begin.")
elif menu == "📋 BOM Manager":
    st.subheader("📋 Bill of Materials (BOM) Manager")

    # Initialize session state for BOM
    if "bom_data" not in st.session_state:
        st.session_state.bom_data = []

    # --- FORM FOR NEW ENTRY ---
    with st.expander("➕ Add New BOM Item"):
        with st.form("add_bom_form"):
            part_name = st.text_input("🔩 Part Name")
            part_id = st.text_input("🔖 Part ID")
            quantity = st.number_input("🔢 Quantity", min_value=1, step=1)
            unit_price = st.number_input("💰 Unit Price ($)", min_value=0.0, step=0.01, format="%.2f")
            supplier = st.text_input("🏢 Supplier")
            delivery_date = st.date_input("🚚 Expected Delivery Date")

            submitted = st.form_submit_button("Add Item")
            if submitted and part_name and part_id:
                st.session_state.bom_data.append({
                    "Part Name": part_name,
                    "Part ID": part_id,
                    "Quantity": quantity,
                    "Unit Price": unit_price,
                    "Supplier": supplier,
                    "Delivery Date": delivery_date,
                    "Total Price": quantity * unit_price
                })
                st.success("✅ Item added successfully!")

    # --- DISPLAY TABLE ---
    if st.session_state.bom_data:
        st.markdown("### 📦 Current BOM List")
        bom_df = pd.DataFrame(st.session_state.bom_data)

        # Update total price column if quantities/prices were edited
        bom_df["Total Price"] = bom_df["Quantity"] * bom_df["Unit Price"]
        total_cost = bom_df["Total Price"].sum()

        edited_df = st.data_editor(bom_df, use_container_width=True, num_rows="dynamic")
        st.session_state.bom_data = edited_df.to_dict("records")

        st.markdown(f"### 💵 **Total Estimated Cost:** ${total_cost:,.2f}")
        st.markdown(f"### 📦 **Total Unique Items:** {len(bom_df)}")

        # Export
        csv = bom_df.to_csv(index=False)
        st.download_button("📥 Download BOM as CSV", csv, file_name="bom_export.csv")

        # Optional: Visualize delivery schedule
        with st.expander("📅 Visualize Delivery Timeline (Gantt Chart)"):
            bom_df["Start"] = pd.to_datetime(datetime.today())
            bom_df["End"] = pd.to_datetime(bom_df["Delivery Date"])
            fig, ax = plt.subplots(figsize=(10, 4))
            for i, row in bom_df.iterrows():
                ax.barh(row["Part Name"], (row["End"] - row["Start"]).days, left=row["Start"], color="orange")
            ax.set_xlabel("Date")
            ax.set_ylabel("Part Name")
            st.pyplot(fig)

    else:
        st.info("No items in your BOM list yet. Add some above to get started.")

elif menu == "🧠 Calculator":
    st.subheader("🧠 Advanced Engineering Calculator")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🧮 Symbolic Math", "🔢 Numeric Solver", "📐 Unit Conversion", "🧾 Matrix Tools", "📊 Plot Functions",
         "📚 Engineering Constants"]
    )

    # Tab 1: Symbolic Math
    with tab1:
        st.markdown("### 🧮 Symbolic Math")
        expr = st.text_input("Enter expression (e.g., x**2 + 3*x + 2)")
        var = st.text_input("Variable (e.g., x)", value="x")
        operation = st.selectbox("Select operation", ["Differentiate", "Integrate", "Simplify", "Expand"])
        if st.button("Solve Symbolically"):
            try:
                x = sp.Symbol(var)
                parsed = sp.sympify(expr)
                if operation == "Differentiate":
                    result = sp.diff(parsed, x)
                elif operation == "Integrate":
                    result = sp.integrate(parsed, x)
                elif operation == "Simplify":
                    result = sp.simplify(parsed)
                else:
                    result = sp.expand(parsed)
                st.success("Result:")
                st.latex(sp.latex(result))
            except Exception as e:
                st.error(f"Error: {e}")

    # Tab 2: Numeric Solver
    with tab2:
        st.markdown("### 🔢 Solve Equations Numerically")
        eqn = st.text_input("Enter equation (e.g., x**2 - 4 = 0)")
        variable = st.text_input("Variable", value="x")
        guess = st.number_input("Initial guess", value=1.0)
        if st.button("Solve Numerically"):
            try:
                x = sp.Symbol(variable)
                lhs, rhs = eqn.split('=')
                f_expr = sp.sympify(lhs) - sp.sympify(rhs)
                f = sp.lambdify(x, f_expr)
                from scipy.optimize import fsolve

                sol = fsolve(f, guess)
                st.success(f"Approximate solution: {sol[0]}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Tab 3: Unit Conversion
    with tab3:
        st.markdown("### 📐 Unit Converter")
        from pint import UnitRegistry

        ureg = UnitRegistry()
        input_qty = st.text_input("Enter quantity with unit (e.g., 10 meter)")
        target_unit = st.text_input("Convert to (e.g., feet)")
        if st.button("Convert"):
            try:
                q = ureg(input_qty)
                converted = q.to(target_unit)
                st.success(f"{q} = {converted}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Tab 4: Matrix Tools
    with tab4:
        st.markdown("### 🧾 Matrix Operations")
        mat_input = st.text_area("Enter matrix (comma-separated rows, e.g. '1 2; 3 4')", value="1 2; 3 4")
        operation = st.selectbox("Select matrix operation", ["Determinant", "Inverse", "Eigenvalues", "Transpose"])
        try:
            matrix = np.array([[float(num) for num in row.split()] for row in mat_input.split(';')])
            if operation == "Determinant":
                result = np.linalg.det(matrix)
            elif operation == "Inverse":
                result = np.linalg.inv(matrix)
            elif operation == "Eigenvalues":
                result = np.linalg.eigvals(matrix)
            elif operation == "Transpose":
                result = matrix.T
            st.write("Result:")
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")

    # Tab 5: Function Plotting
    with tab5:
        st.markdown("### 📊 Function Plotter")
        func = st.text_input("Enter function to plot (e.g., sin(x) * x)")
        domain = st.slider("Domain range for x", -50, 50, (-10, 10))
        if st.button("Plot Graph"):
            try:
                x_vals = np.linspace(domain[0], domain[1], 400)
                y_vals = eval(f"np.{func}")
                plt.figure(figsize=(8, 4))
                plt.plot(x_vals, y_vals, label=func, color='green')
                plt.title("Function Plot")
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Plotting error: {e}")

    # Tab 6: Engineering Constants
    with tab6:
        st.markdown("### 📚 Common Engineering Constants")
        constants = {
            "Gravitational Acceleration (g)": "9.81 m/s²",
            "Speed of Light (c)": "3 x 10⁸ m/s",
            "Planck’s Constant (h)": "6.626 x 10⁻³⁴ Js",
            "Boltzmann Constant (k)": "1.38 x 10⁻²³ J/K",
            "Gas Constant (R)": "8.314 J/(mol·K)",
            "Avogadro's Number": "6.022 x 10²³ 1/mol"
        }
        for key, val in constants.items():
            st.markdown(f"**{key}**: {val}")

elif menu == "📘 Engineering Glossary":
    st.subheader("📘 Engineering Glossary & Reference Hub")
    st.markdown("Search and explore key terms across major engineering disciplines.")

    # Glossary categories
    categories = {
        "Mechanical Engineering": ["Stress", "Strain", "Thermodynamics", "Torque", "Viscosity"],
        "Electrical Engineering": ["Ohm's Law", "Capacitance", "Inductance", "Current", "Voltage"],
        "Civil Engineering": ["Reinforced Concrete", "Load Bearing", "Shear Force", "Bending Moment"],
        "Chemical Engineering": ["Catalysis", "Distillation", "Reaction Rate", "Molarity"],
        "Aerospace Engineering": ["Thrust", "Drag", "Lift", "Mach Number", "Reynolds Number"],
        "Computer Engineering": ["Boolean Algebra", "Flip-Flop", "Cache Memory", "Algorithm"],
        "Data Science": ["Regression", "Clustering", "Overfitting", "Bias-Variance Tradeoff"]
    }

    # Sidebar filtering
    selected_branch = st.selectbox("Select Discipline", list(categories.keys()))
    search_term = st.text_input("🔍 Search for a term (e.g., Torque, Lift)")

    glossary = {
        "Stress": {
            "definition": "Stress is the force applied per unit area within materials.",
            "formula": "σ = F / A",
            "unit": "Pascal (Pa)",
            "field": "Mechanical Engineering"
        },
        "Ohm's Law": {
            "definition": "Defines the relationship between voltage, current, and resistance.",
            "formula": "V = IR",
            "unit": "Volts (V)",
            "field": "Electrical Engineering"
        },
        "Thrust": {
            "definition": "A reaction force that propels a body forward in space or air.",
            "formula": "T = ṁ * (v_e - v_0)",
            "unit": "Newtons (N)",
            "field": "Aerospace Engineering"
        },
        "Lift": {
            "definition": "An aerodynamic force that holds an aircraft in the air.",
            "formula": "L = 0.5 * ρ * v^2 * S * Cl",
            "unit": "Newtons (N)",
            "field": "Aerospace Engineering"
        },
        "Regression": {
            "definition": "A method to model and analyze relationships between variables.",
            "formula": "y = mx + b (simple linear regression)",
            "unit": "N/A",
            "field": "Data Science"
        },
        # Add more terms here as needed...
    }

    # Filter and show terms
    filtered_terms = [term for term in categories[selected_branch] if search_term.lower() in term.lower()]
    for term in filtered_terms:
        if term in glossary:
            st.markdown(f"### 🔹 {term}")
            st.write(f"**Definition:** {glossary[term]['definition']}")
            st.latex(glossary[term]['formula'] if glossary[term]['formula'] != "N/A" else "")
            st.write(f"**Unit:** {glossary[term]['unit']}")
            st.write(f"**Discipline:** {glossary[term]['field']}")

    if not filtered_terms and search_term:
        st.warning("No matching term found. Try a different keyword.")

elif menu == "📎 Notes & Docs":
    st.subheader("📎 Notes & Document Center")
    st.markdown("""
    Manage, create, annotate, and download engineering notes and documents.
    Upload lecture files or type your own, then visualize or export your notes!
    """)

    uploaded_file = st.file_uploader("📤 Upload file (PDF, TXT, DOCX, CSV)", type=["pdf", "txt", "docx", "csv"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == "pdf":
            # Render PDF
            base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        elif file_type == "txt":
            content = uploaded_file.read().decode("utf-8")
            st.text_area("📄 File Content", content, height=300)

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            st.line_chart(df.select_dtypes(include=['float', 'int']))  # Visualize numerics

    st.divider()

    # 📚 CREATE NOTES FROM SCRATCH
    st.subheader("🧾 Create & Save Engineering Notes")
    title = st.text_input("Title of Notes")
    notes = st.text_area("Write your notes below:", height=300)
    file_format = st.selectbox("Choose export format:", ["TXT", "PDF", "Markdown"])

    if st.button("📥 Export Notes"):
        if notes and title:
            if file_format == "TXT":
                st.download_button("Download .txt", notes, file_name=f"{title}.txt")
            elif file_format == "Markdown":
                st.download_button("Download .md", notes, file_name=f"{title}.md")
            elif file_format == "PDF":
                # Generate a temporary PDF
                from fpdf import FPDF

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in notes.split('\n'):
                    pdf.cell(200, 10, txt=line, ln=True)
                pdf_path = f"/tmp/{title}.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as file:
                    st.download_button("Download .pdf", file.read(), file_name=f"{title}.pdf", mime="application/pdf")
        else:
            st.warning("Title and content required.")

    st.divider()

    # 📌 AI SUMMARY (optional if user prefers)
    st.subheader("🧠 Quick Summary of Your Notes")
    if notes:
        with st.spinner("Summarizing your notes..."):

            try:
                summary = summarize(notes)
                st.success("📝 Summary:")
                st.markdown(summary)
            except:
                st.error("Summary failed — text may be too short or complex.")

    # 📊 Visualization of keywords
    if notes:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wordcloud = WordCloud(background_color='white', width=800, height=300).generate(notes)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.subheader("🔍 WordCloud of Your Notes")
        st.pyplot(fig)

elif menu == "🔁 Version Control":
    st.subheader("🔁 Version Control Center")

    # Initialize session state
    if "versions" not in st.session_state:
        st.session_state.versions = []

    # Uploading new version
    st.markdown("### 📥 Upload New Version")
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"])
    version_note = st.text_input("Version Note (e.g., Initial Draft, Final Version v1.2)")

    if st.button("Save Version"):
        if uploaded_file and version_note:
            st.session_state.versions.append({
                "filename": uploaded_file.name,
                "data": uploaded_file.read(),
                "note": version_note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("✅ Version saved successfully!")
        else:
            st.warning("⚠️ Please upload a file and add a version note.")

    # Show all saved versions
    if st.session_state.versions:
        st.markdown("### 🗂 Version History")
        for idx, version in enumerate(reversed(st.session_state.versions), 1):
            with st.expander(f"📄 Version {len(st.session_state.versions) - idx + 1} - {version['note']}"):
                st.write(f"🕒 Timestamp: `{version['timestamp']}`")
                st.write(f"📎 File: `{version['filename']}`")
                if version['filename'].endswith(".txt"):
                    content = version["data"].decode("utf-8")
                    st.code(content, language='text')
                elif version['filename'].endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(version["data"]))
                    st.dataframe(df)
                elif version['filename'].endswith(".pdf"):
                    base64_pdf = base64.b64encode(version["data"]).decode("utf-8")
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400px"></iframe>',
                        unsafe_allow_html=True)

    # Optional: Compare Two Versions
    if len(st.session_state.versions) >= 2:
        st.markdown("### 🔍 Compare Two Versions")
        file_options = [f"{i + 1}: {v['note']}" for i, v in enumerate(st.session_state.versions)]
        version1 = st.selectbox("Select First Version", options=file_options, index=0)
        version2 = st.selectbox("Select Second Version", options=file_options, index=1)

        idx1 = int(version1.split(":")[0]) - 1
        idx2 = int(version2.split(":")[0]) - 1

        data1 = st.session_state.versions[idx1]["data"]
        data2 = st.session_state.versions[idx2]["data"]

        try:
            diff1 = data1.decode()
            diff2 = data2.decode()
            from difflib import ndiff

            diff_result = '\n'.join(ndiff(diff1.splitlines(), diff2.splitlines()))
            st.code(diff_result, language="diff")
        except:
            st.warning("⚠️ Cannot compare these files. Only text-based file comparisons are supported.")

st.markdown("---")
st.markdown("""
  <center>
      🔧 Developed with passion by **Praise Adeyeye**  
      🧠 _"Engineering isn’t just equations — it’s imagination made real."_ 💡  
      <br>
      © 2025 Praise Adeyeye. All rights reserved. 🚀
  </center>
  """, unsafe_allow_html=True)
